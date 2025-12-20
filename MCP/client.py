import asyncio
import threading
import queue

from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat,Client


client = Client(
  host='http://127.0.0.1:11434',
  headers={'x-some-header': 'some-value'}
)

class OllamaMCP:

    def __init__(self, server_params: StdioServerParameters, model_name: str = "qwen3:0.6b"):
        self.server_params = server_params
        self.model_name = model_name
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.initialized = threading.Event()
        self.tools: list[Any] = []
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()
        self.conversation_history = []

    def _run_background(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    self.initialized.set()

                    while True:
                        try:
                            tool_name, arguments = self.request_queue.get(block=False)
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue

                        if tool_name is None:
                            break
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            self.response_queue.put(result)
                        except Exception as e:
                            self.response_queue.put(f"错误: {str(e)}")
        except Exception as e:
            print("MCP会话初始化错误:", str(e))
            self.initialized.set()  # 即使初始化失败也解除等待线程的阻塞
            self.response_queue.put(f"MCP初始化错误: {str(e)}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        发布工具调用请求并等待结果
        """
        if not self.initialized.wait(timeout=30):
            raise TimeoutError("MCP会话未能及时初始化。")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        """
        干净地关闭持久会话
        """
        self.request_queue.put((None, None))
        self.thread.join()
        print("持久MCP会话已关闭。")


    @staticmethod
    def convert_json_type_to_python_type(json_type: str):
        """简单地将JSON类型映射到Python（Pydantic）类型。"""
        if json_type == "integer":
            return (int, ...)
        if json_type == "number":
            return (float, ...)
        if json_type == "string":
            return (str, ...)
        if json_type == "boolean":
            return (bool, ...)
        return (str, ...)

    def create_response_model(self):
        """
        基于获取的工具创建动态Pydantic响应模型
        """
        dynamic_classes = {}
        for tool in self.tools:
            class_name = tool.name.capitalize()
            properties: dict[str, Any] = {}
            for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
                json_type = prop_info.get("type", "string")
                properties[prop_name] = self.convert_json_type_to_python_type(json_type)

            model = create_model(
                class_name,
                __base__=BaseModel,
                __doc__=tool.description,
                **properties,
            )
            dynamic_classes[class_name] = model

        if dynamic_classes:
            all_tools_type = Union[tuple(dynamic_classes.values())]
            Response = create_model(
                "Response",
                __base__=BaseModel,
                __doc__="LLm响应类",
                response=(str, Field(..., description= "向用户确认函数将被调用。")),
                tool=(all_tools_type, Field(
                    ...,
                    description="用于运行和获取魔法输出的工具"
                )),
            )
        else:
            Response = create_model(
                "Response",
                __base__=BaseModel,
                __doc__="LLm响应类",
                response=(str, ...),
                tool=(Optional[Any], Field(None, description="如果不返回None则使用的工具")),
            )
        self.response_model = Response

    async def ollama_chat(self, user_message: str) -> Any:
        """
        使用动态响应模型向Ollama发送消息。
        如果在响应中检测到工具，则使用持久会话调用它。
        """
        # 添加用户消息到对话历史
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # 构建完整的对话上下文
        conversation = [
            {"role": "assistant", "content": f"{[tool.name for tool in self.tools]}"}
        ]
        conversation.extend(self.conversation_history)
        
        if self.response_model is None:
            raise ValueError("响应模型尚未创建。请先调用create_response_model()。")

        # 获取聊天消息格式的JSON模式
        format_schema = self.response_model.model_json_schema()
        print("format_schema", format_schema)

        # 调用Ollama（假定是同步的）并解析响应
        try:
            response = client.chat(
                model=self.model_name,  # 使用指定的模型
                messages=conversation,
                format=format_schema
            )
            print("Ollama响应", response.message.content)
        except Exception as e:
            print(f"Ollama调用失败: {e}")
            # 如果指定模型失败，尝试使用默认模型
            try:
                print(f"尝试使用默认模型 qwen3:0.6b...")
                response = client.chat(
                    model="qwen3:0.6b",
                    messages=conversation,
                    format=format_schema
                )
                print("Ollama响应", response.message.content)
            except Exception as e2:
                print(f"默认模型也失败: {e2}")
                # 返回错误信息
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": f"抱歉，模型调用失败: {str(e2)}"
                })
                return {
                    "type": "error",
                    "response": f"模型调用失败: {str(e2)}"
                }
        
        try:
            response_obj = self.response_model.model_validate_json(response.message.content)
            maybe_tool = response_obj.tool

            if maybe_tool:
                function_name = maybe_tool.__class__.__name__.lower()
                func_args = maybe_tool.model_dump()
                # 使用asyncio.to_thread在线程中调用同步的call_tool方法
                output = await asyncio.to_thread(self.call_tool, function_name, func_args)
                
                # 添加助手回复到对话历史
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": f"工具调用结果: {output}"
                })
                
                return {
                    "type": "tool_result",
                    "response": response_obj.response,
                    "tool_output": output
                }
            else:
                # 添加助手回复到对话历史
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response_obj.response
                })
                
                return {
                    "type": "text_response",
                    "response": response_obj.response
                }
        except Exception as e:
            # 如果JSON解析失败，返回纯文本响应
            print(f"JSON解析失败: {e}")
            text_response = response.message.content
            
            # 添加助手回复到对话历史
            self.conversation_history.append({
                "role": "assistant", 
                "content": text_response
            })
            
            return {
                "type": "text_response",
                "response": text_response
            }

    def get_conversation_history(self):
        """获取对话历史"""
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history.clear()


async def interactive_chat():
    """交互式聊天函数"""
    server_parameters = StdioServerParameters(
        command="python",  # 使用标准Python而不是uv
        args=["server.py"],
        cwd=str(Path.cwd())
    )

    # 创建持久会话，尝试使用baichuanm1，如果失败则使用qwen3:0.6b
    try:
        persistent_session = OllamaMCP(server_parameters, "baichuanm1")
        print("🎯 尝试使用 baichuanm1 模型...")
    except:
        print("⚠️  baichuanm1 模型不可用，使用 qwen3:0.6b 模型...")
        persistent_session = OllamaMCP(server_parameters, "qwen3:0.6b")

    # 等待会话完全初始化
    if persistent_session.initialized.wait(timeout=30):
        print("✅ MCP会话初始化成功！")
        print("🛠️  可用工具:", [tool.name for tool in persistent_session.tools])
        print(f"🤖 使用模型: {persistent_session.model_name}")
    else:
        print("❌ 错误: 初始化超时。")
        return

    # 从获取的工具创建动态响应模型
    persistent_session.create_response_model()
    print("📋 响应模型创建完成")

    print("\n🤖 欢迎使用Ollama MCP聊天机器人！")
    print("💡 输入 'quit' 或 'exit' 退出对话")
    print("💡 输入 'history' 查看对话历史")
    print("💡 输入 'clear' 清空对话历史")
    print("💡 输入 'tools' 查看可用工具")
    print("💡 输入 'model' 查看当前使用的模型")
    print("-" * 50)

    try:
        while True:
            # 获取用户输入
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            elif user_input.lower() == 'history':
                history = persistent_session.get_conversation_history()
                print("\n📚 对话历史:")
                for i, msg in enumerate(history, 1):
                    role_emoji = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{i}. {role_emoji} {msg['role']}: {msg['content']}")
                continue
            elif user_input.lower() == 'clear':
                persistent_session.clear_conversation_history()
                print("🗑️  对话历史已清空")
                continue
            elif user_input.lower() == 'tools':
                print("\n🛠️  可用工具:")
                for i, tool in enumerate(persistent_session.tools, 1):
                    print(f"{i}. {tool.name}: {tool.description}")
                continue
            elif user_input.lower() == 'model':
                print(f"\n🤖 当前使用模型: {persistent_session.model_name}")
                continue
            elif not user_input:
                continue

            print("🤖 助手正在思考...")
            
            try:
                # 调用Ollama并处理响应
                result = await persistent_session.ollama_chat(user_input)
                
                if result["type"] == "tool_result":
                    print(f"🤖 助手: {result['response']}")
                    print(f"🔧 工具输出: {result['tool_output']}")
                elif result["type"] == "error":
                    print(f"❌ 错误: {result['response']}")
                else:
                    print(f"🤖 助手: {result['response']}")
                    
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
                print("🤖 助手: 抱歉，我遇到了一些问题。请重试。")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断，正在退出...")
    finally:
        # 完成后关闭持久会话
        persistent_session.shutdown()


async def main():
    """主函数 - 现在调用交互式聊天"""
    await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())