from openai import OpenAI
from app.util.agent_config import (
    CLINICAL_LANGUAGE_ANALYST_PROMPT,
    NURSE_PROMPT,
    INTELLIGENT_REPORTING_OFFICER_PROMPT,
    NURSE_PROMPT_CALORIES,
    TIJIANBAOGAO_PROMPT,
    METRICS_EXTRACTION_PROMPT,
    get_prompt,
)
from ollama import Client
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama as LangchainOllama
# 从 RAG.py 导入 RAG 相关功能
from app.util.RAG import MedicalRAG, clean_think, medical_rag, all_use_rag, init_medical_rag, get_medical_rag
import os
from tqdm import tqdm
import re
import pandas as pd
from collections import defaultdict
import numpy as np
import json
import copy

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

import asyncio
import threading
import itertools
from collections import defaultdict, deque
import torch
import traceback, linecache
from datetime import datetime
import os
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Rag_device:",device)

# RAG 服务配置
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:5005")


def query_rag_service(question: str, k: int = 3, only_references: bool = True) -> Dict[str, Any]:
    """
    通过 HTTP 请求调用 RAG 服务
    
    Args:
        question: 用户问题
        k: 返回的 top k 结果数量
        only_references: 是否只返回检索内容，不调用大模型（默认 True）
        
    Returns:
        包含 answer 和 references 的字典
        当 only_references=True 时，answer 为空字符串，只返回 references
        
    Raises:
        Exception: 当服务调用失败时抛出异常
    """
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/query",
            json={"question": question, "k": k, "only_references": only_references},
            timeout=200  # 200秒超时
        )
        response.raise_for_status()  # 如果状态码不是 200，会抛出异常
        result = response.json()
        
        if result.get("status") == "success":
            answer = result.get("answer", "")
            references = result.get("references", [])
            
            # 如果只返回检索内容，将检索到的内容组合成文本
            if only_references and references:
                # 将检索到的内容组合成文本
                answer = "\n\n".join([
                    f"【来源：{ref.get('source', 'Unknown')}】\n{ref.get('content', '')}"
                    for ref in references
                ])
            elif not answer or answer.strip() == "":
                print(f"[警告] RAG 服务返回的答案为空，可能数据库中没有相关数据")
                print(f"问题: {question}")
                print(f"参考文献数量: {len(references)}")
            
            return {
                "answer": answer,
                "references": references
            }
        else:
            error_msg = result.get('error', 'Unknown error')
            raise Exception(f"RAG service returned error: {error_msg}")
    except requests.exceptions.Timeout:
        raise Exception(f"RAG service request timeout after 200s. Service URL: {RAG_SERVICE_URL}")
    except requests.exceptions.ConnectionError:
        raise Exception(f"无法连接到 RAG 服务，请确保服务已启动。服务地址: {RAG_SERVICE_URL}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"RAG 服务请求失败: {str(e)}. 服务地址: {RAG_SERVICE_URL}")

class WorkflowLogger:
    """工作流日志记录器，用于记录整个工作流的所有阶段到单个JSON文件"""
    
    def __init__(self, log_dir: str = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/MedicalExaminationAgent/PhysicalExaminationAgent/client/result"):
        self.log_dir = log_dir
        self.workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"workflow_log_{self.workflow_id}.json"
        self.log_path = os.path.join(log_dir, self.log_filename)
        self.log_data = {
            "workflow_id": self.workflow_id,
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "stages": []
        }
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
    
    def _serialize_content(self, content: Any) -> Any:
        """将内容转换为JSON可序列化的格式"""
        try:
            # 处理ReferenceMetrics对象（必须在其他检查之前）
            if content.__class__.__name__ == 'ReferenceMetrics':
                try:
                    return self._serialize_content(content.to_dict())
                except:
                    return {"error": "无法序列化ReferenceMetrics对象"}
            
            # 处理NaN值
            if isinstance(content, float) and (content != content):  # NaN检查
                return None
            
            # 处理numpy的NaN
            if hasattr(content, '__class__') and 'numpy' in str(content.__class__):
                if hasattr(content, 'item'):
                    try:
                        item_value = content.item()
                        if isinstance(item_value, float) and (item_value != item_value):  # NaN检查
                            return None
                        return item_value
                    except:
                        pass
            
            # 如果是基本类型，直接返回
            if isinstance(content, (str, int, float, bool, type(None))):
                return content
            
            # 如果是字典，递归处理
            if isinstance(content, dict):
                return {key: self._serialize_content(value) for key, value in content.items()}
            
            # 如果是列表，递归处理
            if isinstance(content, (list, tuple)):
                return [self._serialize_content(item) for item in content]
            
            # 如果是pandas DataFrame，转换为字典
            if hasattr(content, 'to_dict') and 'DataFrame' in str(type(content)):
                try:
                    df_dict = content.to_dict()
                    # 递归处理DataFrame转换后的字典，确保NaN被处理
                    return self._serialize_content(df_dict)
                except:
                    return str(content)
            
            # 如果是numpy数组，转换为列表
            if hasattr(content, 'tolist'):
                try:
                    array_list = content.tolist()
                    # 递归处理数组转换后的列表，确保NaN被处理
                    return self._serialize_content(array_list)
                except:
                    return str(content)
            
            # 如果有__dict__属性，尝试转换为字典
            if hasattr(content, '__dict__'):
                try:
                    return {key: self._serialize_content(value) for key, value in content.__dict__.items()}
                except:
                    return str(content)
            
            # 如果有__str__方法，使用字符串表示
            if hasattr(content, '__str__'):
                return str(content)
            
            # 最后尝试转换为字符串
            return str(content)
            
        except Exception as e:
            return f"<序列化错误: {str(e)}>"
    
    def _json_default(self, obj):
        """JSON序列化的默认处理器，处理NaN等特殊值"""
        if isinstance(obj, float) and (obj != obj):  # NaN检查
            return None
        if hasattr(obj, '__class__') and 'numpy' in str(obj.__class__):
            if hasattr(obj, 'item'):
                try:
                    item_value = obj.item()
                    if isinstance(item_value, float) and (item_value != item_value):  # NaN检查
                        return None
                    return item_value
                except:
                    pass
        return str(obj)
    
    def log_stage(self, stage_name: str, content: Any):
        """记录工作流阶段"""
        try:
            # 序列化内容
            serialized_content = self._serialize_content(content)
            
            stage_data = {
                "stage_name": stage_name,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "content": serialized_content
            }
            
            self.log_data["stages"].append(stage_data)
            
            # 写入JSON文件
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, ensure_ascii=False, indent=2, default=self._json_default)
                
            print(f"阶段 '{stage_name}' 已记录到: {self.log_path}")
            
        except Exception as e:
            print(f"记录日志时发生错误: {e}")
            traceback.print_exc()
    
    def finalize(self):
        """完成工作流记录"""
        try:
            self.log_data["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_data["total_stages"] = len(self.log_data["stages"])
            
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, ensure_ascii=False, indent=2, default=self._json_default)
                
            print(f"工作流日志已完成: {self.log_path}")
            
        except Exception as e:
            print(f"完成日志记录时发生错误: {e}")
            traceback.print_exc()

def log_workflow_stage(stage_name: str, content: Any, logger: WorkflowLogger = None):
    """
    记录工作流阶段的输出内容到JSON日志文件（兼容性函数）
    
    Args:
        stage_name: 阶段名称
        content: 要记录的内容
        logger: WorkflowLogger实例
    """
    if logger:
        logger.log_stage(stage_name, content)
    else:
        print(f"警告: 未提供logger实例，跳过记录阶段 '{stage_name}'")

def fix_json_format(json_str: str) -> str:
    """
    修复常见的JSON格式问题
    
    Args:
        json_str: 可能包含格式错误的JSON字符串
        
    Returns:
        修复后的JSON字符串
    """
    try:
        # 1. 移除可能的markdown代码块标记
        json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'^```\s*', '', json_str, flags=re.MULTILINE)
        json_str = json_str.strip()
        
        # 2. 修复缺少逗号的情况（对象属性之间）
        # 匹配 "key": value } 或 "key": value ] 或 "key": value { 或 "key": value [ 的情况
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)  # "value" \n "key" -> "value", \n "key"
        json_str = re.sub(r'(\d+)\s*\n\s*"', r'\1,\n"', json_str)  # number \n "key" -> number, \n "key"
        json_str = re.sub(r'(true|false|null)\s*\n\s*"', r'\1,\n"', json_str)  # bool/null \n "key" -> bool/null, \n "key"
        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)  # } \n "key" -> }, \n "key"
        json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)  # ] \n "key" -> ], \n "key"
        
        # 3. 修复数组元素之间缺少逗号的情况
        json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)  # } \n { -> }, \n {
        
        # 4. 移除尾随逗号（JSON标准不允许）
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 5. 修复单引号为双引号
        # 注意：这个操作需要谨慎，因为值中可能包含单引号
        # json_str = json_str.replace("'", '"')
        
        # 6. 移除注释（JSON不支持注释）
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
        
    except Exception as e:
        print(f"JSON格式修复过程出错: {str(e)}")
        return json_str

def extract_metrics_fallback(text: str) -> List[Dict[str, Any]]:
    """
    备用的指标提取策略，使用正则表达式直接从文本中提取指标
    当JSON解析失败时使用
    
    Args:
        text: 包含指标信息的文本
        
    Returns:
        提取到的指标列表
    """
    metrics = []
    
    try:
        # 尝试查找类似 "name": "xxx", "value": "yyy", "unit": "zzz" 的模式
        # 使用正则表达式提取指标信息
        pattern = r'"name"\s*:\s*"([^"]+)"\s*,?\s*"value"\s*:\s*"?([^",}]+)"?\s*,?\s*"unit"\s*:\s*"([^"]*)"'
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            name = match.group(1).strip()
            value = match.group(2).strip()
            unit = match.group(3).strip()
            
            # 跳过空值
            if not name or not value:
                continue
                
            metric = {
                "name": name,
                "value": value,
                "unit": unit,
                "category": "未分类"
            }
            metrics.append(metric)
        
        # 如果上述方法没有找到任何指标，尝试更宽松的模式
        if not metrics:
            # 尝试查找包含中文指标名称和数值的行
            lines = text.split('\n')
            for line in lines:
                # 匹配类似 "身高（cm）: 170" 或 "血糖: 5.5 mmol/L" 的模式
                match = re.search(r'([^:：]+)[：:]\s*([\d.]+)\s*([^\s,，。]*)', line)
                if match:
                    name = match.group(1).strip()
                    value = match.group(2).strip()
                    unit = match.group(3).strip()
                    
                    # 清理指标名称，去除括号内容
                    name = re.sub(r'[（\(][^）\)]*[）\)]', '', name).strip()
                    
                    if name and value:
                        metric = {
                            "name": name,
                            "value": value,
                            "unit": unit,
                            "category": "未分类"
                        }
                        metrics.append(metric)
        
        return metrics
        
    except Exception as e:
        print(f"备用指标提取出错: {str(e)}")
        return []


# ---- 单例：持久事件循环（后台线程） ----
class AsyncRuntime:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._run_loop, daemon=True)
        self._t.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        """在持久 loop 上执行协程，并同步拿结果。"""
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._t.join()

runtime = AsyncRuntime()

# ---- 两个模型 ----
planner_llm = Ollama(model="qwen3:0.6b", request_timeout=60.0)

SYSTEM_PROMPT_PLANNER = "你是一个工具选择助手，负责决定应调用哪些工具。"

# ---- 在同一 loop 中构建 MCP 工具 & Planner Agent ----
def build_planner_agent(mcp_urls: List[str]) -> FunctionAgent:
    async def _build():
        # 在持久 loop 中创建 BasicMCPClient（避免绑定到别的 loop）
        specs = [McpToolSpec(client=BasicMCPClient(url)) for url in mcp_urls]
        all_tools = []
        for spec in specs:
            tools = await spec.to_tool_list_async()
            all_tools.extend(tools)
        return FunctionAgent(
            name="Planner",
            description="Selects which tools to call and with what parameters.",
            tools=all_tools,
            llm=planner_llm,
            system_prompt=SYSTEM_PROMPT_PLANNER,
        )
    return runtime.run(_build())

# ----  在同一 loop 上执行 planner，一次请求一个 Context ----
def run_planner_with_logging_sync(agent: FunctionAgent, user_input: str, language: str = 'zh') -> Tuple[str, List[Dict[str, Any]]]:
    def _out_to_text_and_struct(out):
        """从各种常见结构中尽量提取出 (text, structured)。"""
        if out is None:
            return None, None

        text = None
        structured = getattr(out, "structuredContent", None)

        # 尝试 from out.content
        content = getattr(out, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            t = getattr(first, "text", None)
            if isinstance(first, str):
                t = first
            if t:
                text = t

        # 尝试 raw_output
        raw_output = getattr(out, "raw_output", None)
        if raw_output is not None:
            if structured is None:
                structured = getattr(raw_output, "structuredContent", None)
            ro_content = getattr(raw_output, "content", None)
            if text is None and isinstance(ro_content, list) and ro_content:
                text = getattr(ro_content[0], "text", None)
            if text is None and isinstance(raw_output, str):
                text = raw_output

        # 如果 out 本身是 str
        if text is None and isinstance(out, str):
            text = out

        return text, structured
    async def _runner():
        ctx = Context(agent)
        handler = agent.run(user_input, ctx=ctx)

        tool_logs: List[Dict[str, Any]] = []                 # 顺序视图
        id_to_log: Dict[int, Dict[str, Any]] = {}            # id -> log
        pending_by_tool: Dict[str, deque] = defaultdict(deque)  # 工具名 -> 待完成调用id队列(FIFO)
        return_logs: List[str] = []
        counter = itertools.count(1)  # 自增id

        async for ev in handler.stream_events():
            if isinstance(ev, ToolCall):
                call_id = next(counter)
                args = (getattr(ev, "tool_args", None)
                        or getattr(ev, "tool_kwargs", None)
                        or getattr(ev, "input", None))
                log = {
                    "id": call_id,
                    "tool": ev.tool_name,
                    "args": args,
                    "outputs": [],   # 支持多段
                    "done": False,
                }
                tool_logs.append(log)
                id_to_log[call_id] = log
                pending_by_tool[ev.tool_name].append(call_id)   # 入队（按工具名）
                print(f"→ Calling tool: {ev.tool_name} | id={call_id} | args: {args}")

            elif isinstance(ev, ToolCallResult):
                tool_name = ev.tool_name
                # 取该工具名队列的“当前进行中调用”
                if pending_by_tool[tool_name]:
                    call_id = pending_by_tool[tool_name][0]  # 查看队首但先不弹
                else:
                    # 没有对应的 ToolCall（极端情况）：创建孤儿条目并立刻入队
                    call_id = next(counter)
                    orphan = {
                        "id": call_id, "tool": tool_name, "args": None,
                        "outputs": [], "done": False,
                    }
                    tool_logs.append(orphan)
                    id_to_log[call_id] = orphan
                    pending_by_tool[tool_name].append(call_id)

                out = getattr(ev, "tool_output", None)
                text, structured = _out_to_text_and_struct(out)
                log = id_to_log[call_id]
                log["outputs"].append({"raw": out, "text": text, "structured": structured})

                # 是否为最后一段
                is_last = getattr(ev, "is_last", True)
                if is_last:
                    log["done"] = True
                    # 真正完成才出队，避免多段时提前错位
                    if pending_by_tool[tool_name] and pending_by_tool[tool_name][0] == call_id:
                        pending_by_tool[tool_name].popleft()

                preview = text if text is not None else (structured if structured is not None else "<no content>")
                print(f"✓ Tool {tool_name} (id={call_id}) returned: {preview}")

        resp = await handler

        # 汇总日志
        # 根据语言设置文本标签
        if language == 'en':
            tool_label = "Call Tool"
            args_label = "Input Parameters"
            result_label = "Return Result"
        else:
            tool_label = "调用工具"
            args_label = "传入参数"
            result_label = "返回结果"
        
        for log in tool_logs:
            merged_text = "\n".join([seg["text"] for seg in log["outputs"] if seg.get("text")])
            if not merged_text:
                first_struct = next((seg["structured"] for seg in log["outputs"] if seg.get("structured") is not None), None)
                merged_text = json.dumps(first_struct, ensure_ascii=False) if first_struct is not None else "<no content>"
            return_logs.append(
                f"{tool_label}：{log.get('tool')}\n{args_label}：{log.get('args')}\n{result_label}：{merged_text}"
            )

        return str(resp), return_logs
    return runtime.run(_runner())

def print_exception_with_source(exc, title="Error"):
    print(f"{title}: {type(exc).__name__}: {exc}")
    tb = exc.__traceback__
    while tb and tb.tb_next:
        tb = tb.tb_next
    if tb:
        f = tb.tb_frame
        filename = f.f_code.co_filename
        funcname = f.f_code.co_name
        lineno = tb.tb_lineno
        code_line = linecache.getline(filename, lineno).strip()
        print(f"  at {filename}:{lineno} in {funcname}")
        print(f"  code: {code_line}")
    print("".join(traceback.TracebackException.from_exception(
        exc, capture_locals=True
    ).format(chain=True)))

def chat_with_llm(
    messages: List[Dict[str, str]],
    client: Client,
    # model: str = "jianxiaozhi:latest",
    # model: str = "jianxiaozhi:latest",
    model: str = "qwen3:32b",
    # model: str = "jishi-32B:latest",
    stream: bool = False,
    system_prompt: Optional[str] = None,
    use_rag: bool = True,
    deep_think: bool = False,
    use_mcp: bool = True,
    language: str = 'zh'
) -> Union[str, Dict[str, Any]]:
    """
    统一的函数用于与大模型进行对话
    
    Args:
        messages: 消息列表，格式为 [{"role": "user/assistant", "content": "消息内容"}, ...]
        client: Ollama客户端实例
        model: 使用的模型名称，默认为 qwen3:0.6b
        stream: 是否使用流式输出，默认为False
        system_prompt: 系统提示词，可选
        use_rag: 是否使用RAG系统，默认为True
        
    Returns:
        Union[str, Dict[str, Any]]: 如果stream为True，返回流式响应；否则返回包含RAG响应和LLM响应的字典
    """
    try:
        # 准备消息列表
        chat_messages = []
        rag_response = None
        references = None
        mcp_result = None  # MCP响应结果
        mcp_tool_logs = None  # MCP工具调用日志
        use_tool = False # 是否使用工具
        
        # 如果有系统提示词，添加到消息列表开头
        if system_prompt:
            chat_messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        # 添加用户消息
        chat_messages.extend(messages)

        # 根据deep_think参数给最后一条用户消息添加后缀 (新版本不使用),但为了兼容济世模型使用/think后缀
        if chat_messages and chat_messages[-1]["role"] == "user" and "jishi" in model:
            if deep_think:
                # 思考模式，添加/think后缀（如果没有）
                if not chat_messages[-1]["content"].rstrip().endswith("/think"):
                    chat_messages[-1]["content"] = chat_messages[-1]["content"].rstrip() + "/think"
                deep_think=False # 济世模型后续传入的设为False
            else:
                # 非思考模式，添加/no_think后缀（如果没有）
                if not chat_messages[-1]["content"].rstrip().endswith("/no_think"):
                    chat_messages[-1]["content"] = chat_messages[-1]["content"].rstrip() + "/no_think"

        # 如果启用RAG且最后一条消息是用户消息，尝试获取相关医学知识
        if use_rag and chat_messages and chat_messages[-1]["role"] == "user":
            try:
                user_question = chat_messages[-1]["content"]
                # 通过 Flask RAG 服务查询（默认只返回检索内容，不调用大模型）
                rag_result = query_rag_service(user_question, k=3, only_references=True)
                rag_response = rag_result["answer"]
                references = rag_result["references"]
                print("\n-------------------------------RAG索引完毕--------------------------------\n")
                
                # 根据语言设置不同的前缀
                if language == 'en':
                    rag_prefix = "Relevant Medical Knowledge:"
                else:
                    rag_prefix = "相关医学知识："
                
                # 将RAG结果添加到系统提示中
                if system_prompt:
                    chat_messages[0]["content"] += f"\n\n{rag_prefix}\n{rag_response}"
                else:
                    chat_messages.insert(0, {
                        "role": "system",
                        "content": f"{rag_prefix}\n{rag_response}"
                    })
            except Exception as e:
                print(f"RAG error: {str(e)}")
        # === MCP Tool Calling ===
        if use_mcp:
            planner_agent = build_planner_agent([
                "http://127.0.0.1:9001/mcp",  # time
                "http://127.0.0.1:9000/mcp",  # calc
                "https://mcp.map.baidu.com/mcp?ak=hntNzrvIxDafXo4DKc8i7quf48AUROwn", # baidumap
                "http://127.0.0.1:9007/sse", # nexonco
                "http://127.0.0.1:9008/sse" # HowToCook
            ])

            planner_result, tool_logs = run_planner_with_logging_sync(planner_agent, chat_messages[-1]["content"], language=language)
            # 保存MCP响应用于前端显示
            mcp_result = planner_result
            mcp_tool_logs = tool_logs
            if tool_logs:  # 只有真的调用了工具才插入
                use_tool = True
                print("tool_logs:",tool_logs)
                
                # 根据语言设置不同的前缀
                if language == 'en':
                    tool_prefix = "Tools have been successfully called. Tool call results:"
                else:
                    tool_prefix = "已成功调用工具，工具调用结果："
                
                if system_prompt:
                    chat_messages[0]["content"] += f"\n\n{tool_prefix}\n{tool_logs}"
                else:
                    chat_messages.insert(0, {
                        "role": "system",
                        "content": f"{tool_prefix}\n{tool_logs}"
                    })
        print("CHAT_MESSAGES:",chat_messages)
        print("DEEP_THINK:",deep_think)
        print("STREAM:",stream)
        print("MODEL:",model)
        print("USE_RAG:",use_rag)
        print("USE_MCP:",use_mcp)
        # 调用Ollama API
        response = client.chat(
            model=model,
            messages=chat_messages,
            stream=stream,
            think=deep_think,
            keep_alive=-1
            
        )
        # if deep_think:
        #     print("RESPONSE:",response.message.thinking)
        # else:
        #     print("RESPONSE:",response)
        print("RESPONSE:",response)
        
        if stream:
            # 返回流式响应对象，包含MCP响应
            if use_rag and use_mcp and use_tool:
                return response, references, {"result": mcp_result, "tool_logs": mcp_tool_logs}
            elif use_rag:
                return response, references, None
            elif use_mcp and use_tool:
                return response, None, {"result": mcp_result, "tool_logs": mcp_tool_logs}
            else:
                return response, None, None
        else:
            llm_response = response.message.content.strip()
            if use_rag and use_mcp and use_tool:
                return {
                    "rag_response": rag_response,
                    "llm_response": llm_response,
                    "references": references,
                    "mcp_result": mcp_result,
                    "mcp_tool_logs": mcp_tool_logs
                }
            elif use_rag:
                return {
                    "rag_response": rag_response,
                    "llm_response": llm_response,
                    "references": references
                }
            elif use_mcp and use_tool:
                return {
                    "llm_response": llm_response,
                    "mcp_result": mcp_result,
                    "mcp_tool_logs": mcp_tool_logs
                }
            else:
                return llm_response
            
    except Exception as e:
        print(f"LLM chat error: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
        print_exception_with_source(e, title="LLM chat error")
        raise
# 判断是否调用体检报告分析工作流
def is_call_report_workflow(dialogue: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> bool:
    """
    判断是否调用体检报告分析工作流
    
    Args:
        dialogue: 用户输入
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    """
    try:
        prompt_template = get_prompt('TIJIANBAOGAO_PROMPT', language)
        prompt = prompt_template.format(user_input=dialogue)
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_rag=False,
            use_mcp=False
        )
        
        # 获取LLM响应
        if isinstance(result, dict):
            result = result["llm_response"]
        
        # 提取数字并判断
        return extract_number_and_judge(result)
        
    except Exception as e:
        print(f"Error in is_call_report_workflow: {str(e)}")
        return False

def extract_number_and_judge(response: str) -> bool:
    """
    从响应中提取数字并判断是否需要调用体检报告工作流
    
    Args:
        response: LLM的响应内容
        
    Returns:
        bool: True表示需要调用体检报告工作流，False表示不需要
    """
    try:
        # 清理响应内容
        cleaned_response = clean_think(response)
        
        # 移除多余的空格、换行符和其他字符
        cleaned_response = cleaned_response.replace(" ", "").replace("\n", "").replace("'", "").replace('"', "").replace("```", "").replace("python", "")
        
        # 使用正则表达式提取数字
        import re
        numbers = re.findall(r'\d+', cleaned_response)
        
        if numbers:
            # 如果找到数字，取第一个数字进行判断
            first_number = int(numbers[0])
            return first_number == 1
        else:
            # 如果没有找到数字，检查是否包含关键词
            keywords = ['是', '需要', '调用', '体检', '报告', 'yes', 'true', '1']
            return any(keyword in cleaned_response.lower() for keyword in keywords)
            
    except Exception as e:
        print(f"Error in extract_number_and_judge: {str(e)}")
        return False
    

def analyze_dialogue(dialogue: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> int:
    """
    使用临床语言分析师分析对话内容，判断是否需要调用风险评估模型
    
    Args:
        dialogue: 用户对话内容
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    
    Returns:
        int: 风险模型编号（-1表示无需调用任何模型）
    """
    try:
        # 准备提示词
        prompt_template = get_prompt('CLINICAL_LANGUAGE_ANALYST_PROMPT', language)
        prompt = prompt_template.format(dialogue=dialogue)
        
        # 使用统一的对话函数
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_rag=False,
            use_mcp=False
        )
        print("临床语言分析师：",result)
        # 获取LLM响应
        if isinstance(result, dict):
            result = result["llm_response"]
        
        # 移除<think>标签及其内容
        if "<think>" in result and "</think>" in result:
            result = result.split("</think>")[-1].strip()
        
        # 清理结果中的其他字符
        result = result.replace(" ", "").replace("\n", "").replace("'", "").replace('"', "").replace("```", "").replace("python", "")
        
        # 提取所有数字
        import re
        numbers = re.findall(r'-?\d+', result) # 匹配正负整数
        
        if numbers:
            # 如果只有一个数字，直接返回
            if len(numbers) == 1:
                return int(numbers[0])
            # 如果有多个数字，返回最大的数字（通常表示最高风险等级）
            return max(int(num) for num in numbers)
        return -1  # 如果没有找到数字，返回-1
        
    except Exception as e:
        print(f"Clinical analysis error: {str(e)}")
        return -1  # 发生错误时返回-1

def get_nurse_response(model_name: str, user_info: str, form_data: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> Dict[str, str]:
    """
    获取护士对用户健康评估的回应建议
    
    Args:
        model_name: 使用的健康风险评估模型名称
        user_info: 用户的具体情况信息
        form_data: 表单数据
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的Ollama模型名称，默认为 qwen3:32b
    
    Returns:
        Dict[str, str]: 包含RAG响应和护士健康建议的字典
    """
    try:
        if model_name == 'calories':
            prompt_template = get_prompt('NURSE_PROMPT_CALORIES', language)
            prompt = prompt_template.format(user_info=user_info, form_data=form_data)
        else:
            prompt_template = get_prompt('NURSE_PROMPT', language)
            prompt = prompt_template.format(model_name=model_name, user_info=user_info, form_data=form_data)
        # 准备提示词
        # prompt = NURSE_PROMPT.format(model_name=model_name, user_info=user_info, form_data=form_data)
        
        # 使用统一的对话函数
        return chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_rag=False,
            use_mcp=False
        )
        
    except Exception as e:
        print(f"Nurse response error: {str(e)}")
        return {
            "rag_response": None,
            "llm_response": "抱歉，暂时无法提供健康建议。请稍后再试。"
        }

def generate_health_report(dialogue: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> Dict[str, str]:
    """
    根据用户与医生的对话内容生成健康体检报告
    
    Args:
        dialogue: 用户与医生的对话内容
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    
    Returns:
        Dict[str, str]: 包含RAG响应和格式化的健康体检报告的字典
    """
    try:
        # 准备提示词
        prompt_template = get_prompt('INTELLIGENT_REPORTING_OFFICER_PROMPT', language)
        prompt = prompt_template.format(dialogue=dialogue)
        
        # 使用统一的对话函数
        return chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_rag=False,
            use_mcp=False
        )
        
    except Exception as e:
        print(f"Health report generation error: {str(e)}")
        return {
            "rag_response": None,
            "llm_response": "抱歉，暂时无法生成健康体检报告。请稍后再试。"
        }

def generate_follow_up_questions(question: str, answer: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> List[str]:
    """
    根据用户问题和AI回答生成后续追问建议
    
    Args:
        question: 用户的原始问题
        answer: AI的回答内容
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    
    Returns:
        List[str]: 最多3条追问建议的列表
    """
    try:
        # 准备提示词
        prompt_template = get_prompt('PROBLEM_WIZARD_PROMPT', language)
        prompt = prompt_template.format(question=question, answer=answer)
        
        # 使用统一的对话函数
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_rag=False,
            use_mcp=False
        )
        
        # 获取LLM响应
        if isinstance(result, dict):
            response = result["llm_response"]
        else:
            response = result
        
        # 解析响应，提取问题建议
        questions = []
        response = clean_think(response)
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # 去除markdown加粗格式
            line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            # 匹配数字开头的行（如 "1. 问题1"）
            if re.match(r'^\d+\.', line):
                # 移除数字和点号，提取问题内容
                question_text = re.sub(r'^\d+\.\s*', '', line)
                if question_text and (question_text != "<think>" and question_text != "</think>"):
                    questions.append(question_text)
            # 也匹配其他可能的格式
            elif line and not line.startswith('用户问题：') and not line.startswith('你的回答：') and (line != "<think>" and line != "</think>"):
                questions.append(line)
        print("引导问题：",questions)
        # 限制最多3个问题
        return questions[:3]
        
    except Exception as e:
        print(f"Follow-up questions generation error: {str(e)}")
        return []

class ReferenceMetrics:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.index_map = defaultdict(list)
        self.duplicates = {'指标名': [], '英文名': [], '英文简称': []}
        # 存储原始键名到标准化键名的映射（用于英文名称和简称的大小写不敏感匹配）
        self.key_normalization = {}

        # 构建中英文索引映射（仅指标名、英文名、简称、别称，不包括类别）
        # 避免类别匹配导致返回多行DataFrame
        for idx, row in df.iterrows():
            # 只为具体指标建立索引，不包括类别
            metric_name = str(row['指标名']).strip()
            eng_name = str(row['英文名']).strip()
            abbr = str(row['英文简称']).strip()
            
            # 处理别称列（如果存在）
            aliases = ""
            if '别称' in df.columns:
                aliases = str(row.get('别称', '')).strip()
            
            # 跳过空值
            if metric_name and metric_name != 'nan':
                # 中文名称保持原样
                self.index_map[metric_name].append(idx)
                if len(self.index_map[metric_name]) > 1:
                    self.duplicates['指标名'].append(metric_name)
                    
            if eng_name and eng_name != 'nan':
                # 英文名称：同时存储原始值和大写值，实现大小写不敏感匹配
                self.index_map[eng_name].append(idx)
                eng_name_upper = eng_name.upper()
                # 总是存储大写版本（即使原始值已经是大写），确保所有大小写变体都能匹配
                if eng_name_upper not in self.index_map or idx not in self.index_map[eng_name_upper]:
                    self.index_map[eng_name_upper].append(idx)
                if eng_name_upper != eng_name:
                    # 记录大写版本到原始版本的映射
                    self.key_normalization[eng_name_upper] = eng_name
                if len(self.index_map[eng_name]) > 1 and eng_name not in self.duplicates['英文名']:
                    self.duplicates['英文名'].append(eng_name)
                    
            if abbr and abbr != 'nan':
                # 英文简称：同时存储原始值和大写值，实现大小写不敏感匹配
                self.index_map[abbr].append(idx)
                abbr_upper = abbr.upper()
                # 总是存储大写版本（即使原始值已经是大写），确保所有大小写变体都能匹配
                if abbr_upper not in self.index_map or idx not in self.index_map[abbr_upper]:
                    self.index_map[abbr_upper].append(idx)
                if abbr_upper != abbr:
                    # 记录大写版本到原始版本的映射
                    self.key_normalization[abbr_upper] = abbr
                if len(self.index_map[abbr]) > 1 and abbr not in self.duplicates['英文简称']:
                    self.duplicates['英文简称'].append(abbr)
            
            # 处理别称列：别称可能包含多个值（用逗号分隔），需要分割并分别索引
            if aliases and aliases != 'nan' and aliases != '':
                # 分割别称（支持逗号、分号、空格分隔）
                alias_list = re.split(r'[,;，；\s]+', aliases)
                for alias in alias_list:
                    alias = alias.strip()
                    if alias and alias != '':
                        # 别称也支持大小写不敏感匹配（如果是英文）
                        self.index_map[alias].append(idx)
                        # 检查是否是英文（不包含中文字符）
                        if not any('\u4e00' <= char <= '\u9fff' for char in alias):
                            alias_upper = alias.upper()
                            # 存储大写版本以实现大小写不敏感匹配
                            if alias_upper not in self.index_map or idx not in self.index_map[alias_upper]:
                                self.index_map[alias_upper].append(idx)
                            if alias_upper != alias:
                                # 记录大写版本到原始版本的映射
                                self.key_normalization[alias_upper] = alias
        
        # 打印重复警告
        if any(self.duplicates.values()):
            print("⚠️ 参考数据中发现重复项:")
            for key, values in self.duplicates.items():
                if values:
                    print(f"  {key}: {len(values)} 个重复 (如: {', '.join(values[:3])}...)")
                    

    def __getitem__(self, key):
        # 首先尝试直接匹配（保持向后兼容）
        # 支持匹配：指标名、英文名、英文简称、别称
        lookup_key = key
        if key not in self.index_map:
            # 如果直接匹配失败，尝试大小写不敏感匹配（对英文名称、简称和别称）
            # 检查是否是英文（不包含中文字符）
            if not any('\u4e00' <= char <= '\u9fff' for char in str(key)):
                key_upper = str(key).upper()
                if key_upper in self.index_map:
                    # 如果找到了大写版本，使用它来查找
                    lookup_key = key_upper
                else:
                    raise KeyError(f"找不到对应指标: {key}")
            else:
                # 中文名称保持原样匹配
                raise KeyError(f"找不到对应指标: {key}")
        
        indices = self.index_map[lookup_key]
        
        # 如果匹配到多行，发出警告并只返回第一行
        if len(indices) > 1:
            print(f"⚠️ 警告: 键 '{key}' 匹配到 {len(indices)} 行数据，只使用第一行")
        
        # 总是返回字典（第一行数据）
        idx = indices[0]
        row_dict = self.df.loc[idx].to_dict()
        
        # 将Series或其他复杂类型转换为标量值
        for k, v in row_dict.items():
            if hasattr(v, 'item'):  # pandas scalar
                row_dict[k] = v.item() if pd.notna(v) else ""
            elif pd.isna(v):  # NaN values
                row_dict[k] = ""
        
        return row_dict
    
    def __bool__(self):
        """支持布尔判断，用于 if not reference_data: 这样的条件"""
        return self.df is not None and not self.df.empty
    
    def __len__(self):
        """返回参考指标的数量"""
        return len(self.df)
    
    def to_dict(self):
        """转换为可序列化的字典格式，用于日志记录"""
        sample_data = []
        if len(self.df) > 0:
            # 获取前3行数据
            sample_df = self.df.head(3)
            for _, row in sample_df.iterrows():
                row_dict = {}
                for col, val in row.items():
                    # 处理NaN值和pandas scalar
                    if pd.isna(val):
                        row_dict[col] = None
                    elif hasattr(val, 'item'):
                        row_dict[col] = val.item()
                    else:
                        row_dict[col] = val
                sample_data.append(row_dict)
        
        return {
            "total_metrics": len(self.df),
            "columns": list(self.df.columns),
            "sample_data": sample_data
        }
def load_reference_metrics(csv_path: str, language: str = 'zh') -> Optional[ReferenceMetrics]:
    """
    加载参考指标数据
    
    Args:
        csv_path: CSV文件路径
        language: 语言类型，'zh' 表示中文，'en' 表示英文
        
    Returns:
        ReferenceMetrics对象，如果加载失败则返回None
    """
    try:
        # 使用更宽松的CSV解析选项，处理字段中包含逗号的情况
        # 尝试使用标准解析
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.ParserError:
            # 如果标准解析失败，使用更宽松的选项
            df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
        
        if df.empty:
            print(f"警告: CSV文件 '{csv_path}' 为空")
            return None
        
        # 根据语言检查必需的列
        if language == 'en':
            # 英文CSV的列名映射
            column_mapping = {
                'Metric Name (CN)': '指标名',
                'English Name': '英文名',
                'English Abbreviation': '英文简称',
                'Category': '类别',
                'Lower Bound': '左边界',
                'Upper Bound': '右边界',
                'Unit': '单位',
                'Professional Interpretation': '专业解读',
                'Recommendations': '建议',
                'Applicable Scope': '适用范围',
                'Health Interpretation': '健康解读',
                'Aliases': '别称'  # 添加别称列的映射
            }
            # 重命名列以匹配内部使用的列名
            df = df.rename(columns=column_mapping)
            required_columns = ['指标名', '英文名', '英文简称']
        else:
            # 中文CSV使用原始列名
            required_columns = ['指标名', '英文名', '英文简称']
            # 如果中文CSV也有别称列，保持原样（列名可能是'别称'）
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误: CSV文件缺少必需的列: {missing_columns}")
            return None
            
        return ReferenceMetrics(df)
    except FileNotFoundError:
        print(f"错误: 找不到CSV文件 '{csv_path}'")
        return None
    except pd.errors.EmptyDataError:
        print(f"错误: CSV文件 '{csv_path}' 为空")
        return None
    except Exception as e:
        print(f"错误: 加载CSV文件 '{csv_path}' 失败: {str(e)}")
        return None

def convert_unit(value, from_unit: str, to_unit: str):
    """
    转换医学指标的单位，包括血糖、血脂、血红蛋白、身高、体重等。
    """
    # 处理pandas Series或其他对象
    if hasattr(from_unit, 'item'):
        from_unit = str(from_unit.item()) if pd.notna(from_unit) else ""
    if hasattr(to_unit, 'item'):
        to_unit = str(to_unit.item()) if pd.notna(to_unit) else ""
    
    # 转换为字符串类型
    from_unit = str(from_unit) if from_unit else ""
    to_unit = str(to_unit) if to_unit else ""
    
    # 处理非数值类型的值（如"阴性"、"阳性"等）
    if not isinstance(value, (int, float)):
        try:
            # 尝试转换为浮点数
            value = float(value)
        except (ValueError, TypeError):
            # 如果无法转换，返回原值（可能是字符串）
            print(f"警告: 无法将值 '{value}' 转换为数值，跳过单位转换")
            return value
    
    # 清洗格式
    from_unit = from_unit.strip().lower().replace("^", "").replace("×", "x")
    to_unit = to_unit.strip().lower().replace("^", "").replace("×", "x")

    # 标准别名映射
    alias_map = {
        "×10^9/l": "x109/l",
        "×10⁹/l": "x109/l",
        "μmol/l": "umol/l",
        "µmol/l": "umol/l",
        "公斤": "kg",
        "斤": "jin",
        "厘米": "cm",
        "米": "m",
        "磅": "lbs"
    }

    from_unit_std = alias_map.get(from_unit, from_unit)
    to_unit_std = alias_map.get(to_unit, to_unit)

    # ✅ 判断单位是否一致（应放在标准化之后）
    if from_unit_std == to_unit_std:
        return round(float(value), 2)

    unit_conversion = {
        # 血糖
        ("mmol/l", "mg/dl"): lambda v: v * 18.0,
        ("mg/dl", "mmol/l"): lambda v: v / 18.0,

        # 胆固醇
        ("mmol/l", "mg/dl (cholesterol)"): lambda v: v * 38.67,
        ("mg/dl", "mmol/l (cholesterol)"): lambda v: v / 38.67,

        # 肌酐
        ("umol/l", "mg/dl"): lambda v: v * 0.0113,
        ("mg/dl", "umol/l"): lambda v: v / 0.0113,

        # 血红蛋白
        ("g/l", "g/dl"): lambda v: v / 10.0,
        ("g/dl", "g/l"): lambda v: v * 10.0,

        # 血小板
        ("x109/l", "/l"): lambda v: v * 1e9,
        ("/l", "x109/l"): lambda v: v / 1e9,

        # 身高
        ("cm", "m"): lambda v: v / 100,
        ("m", "cm"): lambda v: v * 100,

        # 体重
        ("kg", "jin"): lambda v: v * 2,
        ("jin", "kg"): lambda v: v / 2,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
    }

    key = (from_unit_std, to_unit_std)

    if key not in unit_conversion:
        print(f"警告: 暂不支持从 {from_unit} 转换为 {to_unit}，返回原值")
        try:
            return round(float(value), 2)
        except (ValueError, TypeError):
            return value

    try:
        return round(unit_conversion[key](float(value)), 2)
    except (ValueError, TypeError):
        print(f"警告: 单位转换时无法处理值 '{value}'，返回原值")
        return value


def translate_reference_range(ref_range: str, language: str = 'zh') -> str:
    """
    翻译参考范围字符串，将中文翻译为英文或保持中文
    
    Args:
        ref_range: 参考范围字符串
        language: 目标语言，'zh' 或 'en'
    
    Returns:
        翻译后的参考范围字符串
    """
    if language == 'zh':
        return ref_range
    
    # 英文翻译映射
    translations = {
        '暂无参考范围': 'No reference range available',
        '无-无 阴性': 'Negative',
        '无-无': 'N/A',
        '阴性': 'Negative',
        '阳性': 'Positive',
        '参考范围': 'Reference range'
    }
    
    # 如果整个字符串在映射中，直接返回翻译
    if ref_range in translations:
        return translations[ref_range]
    
    # 替换部分中文文本
    translated = ref_range
    for zh_text, en_text in translations.items():
        translated = translated.replace(zh_text, en_text)
    
    return translated


def analyze_uploaded_metrics(extracted_metrics: Dict[str, Any], reference_data, language: str = 'zh') -> Dict[str, Any]:
    # 深拷贝，避免修改原始数据
    metrics_data = copy.deepcopy(extracted_metrics)
    metrics = metrics_data.get("metrics", [])
    
    analysis_results = []
    metrics_name = [m.get("name", "") for m in metrics]
    
    print("---------------------------------指标名称---------------------------------\n", metrics_name)

    # 定义中英文可能的别名
    height_aliases = {"身高", "Height", "height"}
    weight_aliases = {"体重", "Weight", "weight"}

    height_idx = weight_idx = None

    for idx, name in enumerate(metrics_name):
        if name in height_aliases:
            height_idx = idx
        elif name in weight_aliases:
            weight_idx = idx

    # 如果都找到，计算 BMI
    if height_idx is not None and weight_idx is not None:
        try:
            height_value = float(metrics[height_idx].get("value", ""))
            weight_value = float(metrics[weight_idx].get("value", ""))
            if height_value > 0:
                bmi = weight_value / (height_value / 100) ** 2
                metrics.append({
                    "name": "BMI",
                    "value": round(bmi, 2),
                    "unit": "kg/m²",
                    "category": "身体指标"
                })
        except (ValueError, ZeroDivisionError):
            pass  # 数据非法，跳过

    print("---------------------------------指标数据---------------------------------\n", metrics)

    for metric in metrics:
        metric_name = metric.get("name", "")
        metric_value = metric.get("value", "")
        metric_unit = metric.get("unit", "")
        metric_category = metric.get("category", "")

        try:
            # reference_data[metric_name] 现在总是返回字典
            ref = reference_data[metric_name]
            lower_bound = ref.get("左边界", "")
            upper_bound = ref.get("右边界", "")
            ref_unit = ref.get("单位", "")
            professional_interpretation = ref.get("专业解读", "")
            suggestion = ref.get("建议", "")
            applicable = ref.get("适用范围", "")
            health_interpretation = ref.get("健康解读", "")
            
            # 确保所有值都是标量（防止Series对象）
            def ensure_scalar(val):
                if hasattr(val, 'item'):
                    return val.item() if pd.notna(val) else ""
                elif pd.isna(val):
                    return ""
                return val
            
            lower_bound = ensure_scalar(lower_bound)
            upper_bound = ensure_scalar(upper_bound)
            ref_unit = ensure_scalar(ref_unit)
            professional_interpretation = ensure_scalar(professional_interpretation)
            suggestion = ensure_scalar(suggestion)
            applicable = ensure_scalar(applicable)
            health_interpretation = ensure_scalar(health_interpretation)

            # 单位转换（确保你有这个函数）
            try:
                metric_value = convert_unit(metric_value, metric_unit, ref_unit)
                metric_unit = ref_unit
            except Exception as e:
                print(f"警告: 单位转换失败 '{metric_name}': {e}，使用原值")
                # 保持原值不变
        except KeyError:
            lower_bound = upper_bound = ref_unit = professional_interpretation = suggestion = applicable = health_interpretation = ""
        
        # 构造参考范围字符串
        if str(lower_bound) != "" and str(upper_bound) != "":
            metric_reference_range = f"{lower_bound}-{upper_bound} {ref_unit}"
        else:
            metric_reference_range = "暂无参考范围" if language == 'zh' else "No reference range available"

        # 打印分析结果
        # print(f"{metric_name}：为 {metric_value}{metric_unit}，属于 {metric_category} 类指标。")
        # print(f"参考范围为：{metric_reference_range}。")
        # if health_interpretation:
        #     print(f"健康解读：{health_interpretation}")
        # if suggestion:
        #     print(f"诊疗建议：{suggestion}")
        # if applicable:
        #     print(f"适用范围：{applicable}")
        # if professional_interpretation:
        #     print(f"专业解读：{professional_interpretation}")
        # print()

        analysis_results.append({
            "name": metric_name,
            "value": metric_value,
            "unit": metric_unit,
            "category": metric_category,
            "reference_range": metric_reference_range,
            "health_interpretation": health_interpretation,
            "suggestion": suggestion,
            "applicable": applicable,
            "professional_interpretation": professional_interpretation
        })

    return {"analysis": analysis_results}


def extract_metrics_from_dialogue(dialogue: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> Dict[str, Any]:
    """
    从用户与医生的对话中提取体检指标数据
    
    Args:
        dialogue: 用户与医生的对话内容
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    
    Returns:
        Dict[str, Any]: 包含提取的指标数据的字典，格式如下：
        {
            "metrics": [
                {
                    "name": "指标名称",
                    "value": "数值",
                    "unit": "单位",
                    "reference_range": "参考范围",
                    "status": "状态",
                    "category": "指标类别"
                }
            ],
            "metrics_count": 指标个数（整数）,
            "extraction_confidence": "提取置信度",
            "missing_info": "缺失信息"
        }
    """
    try:
        import json
        
        # 准备提示词
        prompt_template = get_prompt('METRICS_EXTRACTION_PROMPT', language)
        prompt = prompt_template.format(dialogue=dialogue)
        
        # 使用统一的对话函数
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_rag=False,
            use_mcp=False,
            deep_think=True
        )
        print(result)
        # 获取LLM响应
        if isinstance(result, dict):
            response = result["llm_response"]
        else:
            response = result
        
        # 清理响应内容
        response = clean_think(response)
        
        # 尝试解析JSON
        try:
            # 查找JSON内容
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                
                # 尝试修复常见的JSON格式问题
                json_content = fix_json_format(json_content)
                
                extracted_data = json.loads(json_content)
                
                # 验证数据结构
                if isinstance(extracted_data, dict) and "metrics" in extracted_data:
                    # 清理指标名称，去除括号及括号内的内容
                    for metric in extracted_data.get('metrics', []):
                        if 'name' in metric and metric['name']:
                            # 使用正则表达式去除中英文括号及其内容，并去除前后空格
                            original_name = metric['name']
                            cleaned_name = re.sub(r'[（\(][^）\)]*[）\)]', '', original_name).strip()
                            metric['name'] = cleaned_name
                            if original_name != cleaned_name:
                                print(f"指标名称清理: '{original_name}' -> '{cleaned_name}'")
                    
                    # 添加指标个数到返回结果中
                    metrics_count = len(extracted_data.get('metrics', []))
                    extracted_data['metrics_count'] = metrics_count
                    print(f"成功提取到 {metrics_count} 个指标")
                    return extracted_data
                else:
                    print("提取的数据格式不正确")
                    return {
                        "metrics": [],
                        "metrics_count": 0,
                        "extraction_confidence": "低",
                        "missing_info": "数据格式错误"
                    }
            else:
                print("未找到有效的JSON内容")
                return {
                    "metrics": [],
                    "metrics_count": 0,
                    "extraction_confidence": "低",
                    "missing_info": "未找到有效的JSON数据"
                }
                
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            print(f"错误位置: 行 {e.lineno}, 列 {e.colno}")
            print(f"原始响应（前500字符）: {response[:500]}")
            print(f"尝试修复后的JSON（前500字符）: {json_content[:500] if 'json_content' in locals() else 'N/A'}")
            
            # 尝试备用解析策略：使用更宽松的方法
            try:
                print("尝试备用解析策略...")
                # 尝试逐行提取指标信息（即使JSON格式有问题）
                fallback_metrics = extract_metrics_fallback(response)
                if fallback_metrics:
                    fallback_count = len(fallback_metrics)
                    print(f"备用策略成功提取到 {fallback_count} 个指标")
                    return {
                        "metrics": fallback_metrics,
                        "metrics_count": fallback_count,
                        "extraction_confidence": "中",
                        "missing_info": f"JSON格式有误，使用备用解析策略提取"
                    }
            except Exception as fallback_error:
                print(f"备用解析策略也失败: {str(fallback_error)}")
            
            return {
                "metrics": [],
                "metrics_count": 0,
                "extraction_confidence": "低",
                "missing_info": f"JSON解析失败 (行{e.lineno},列{e.colno}): {str(e)}"
            }
            
    except Exception as e:
        print(f"指标提取错误: {str(e)}")
        return {
            "metrics": [],
            "metrics_count": 0,
            "extraction_confidence": "低",
            "missing_info": f"提取过程出错: {str(e)}"
        }
def generate_report(data):
    """
    生成体检报告，先输出正常指标，再输出异常指标
    """
    def is_abnormal(value, reference_range):
        """
        判断指标是否异常
        通过比较value和reference_range来判断
        """
        # 将value转换为字符串以便检查
        value_str = str(value).strip().lower()
        
        # 优先检查是否为"阳性"（包括各种可能的表示方式）
        # 注意：即使参考范围为"暂无参考范围"，阳性也应该被识别为异常
        positive_keywords = ["阳性", "positive", "+", "pos", "异常", "abnormal"]
        if any(keyword in value_str for keyword in positive_keywords):
            return True
        
        # 如果没有参考范围，无法进行数值比较，返回False
        if not reference_range or reference_range == "暂无参考范围":
            return False
        
        try:
            # 尝试转换value为浮点数
            metric_value = float(value)
            
            # 解析参考范围，格式如 "125-350 ×10^9/L" 或 "3.9-6.1 mmol/L"
            range_str = reference_range.split()[0] if ' ' in reference_range else reference_range
            
            # 处理不同格式的参考范围
            if '-' in range_str:
                parts = range_str.split('-')
                if len(parts) == 2:
                    lower = float(parts[0])
                    upper = float(parts[1])
                    # 判断是否在正常范围内
                    return metric_value < lower or metric_value > upper
            elif '<' in range_str or '>' in range_str:
                # 处理 "<90/60" 或 ">140/90" 这类情况
                return True  # 这类通常表示异常标准
                
        except (ValueError, IndexError, AttributeError):
            # 如果无法解析，默认认为是正常的（避免误报）
            pass
        
        return False
    
    def format_metric_simple(metric):
        """格式化正常指标的输出（简洁版）"""
        name = metric["name"]
        value = metric["value"]
        unit = metric["unit"]
        ref = metric["reference_range"] or "暂无参考范围"
        category = metric["category"]
        
        # 正常指标只显示基本信息，不显示解读和建议
        return f"{name}为{value}{unit}，参考范围：{ref}，属于 {category} 类指标。"
    
    def format_metric_detailed(metric):
        """格式化异常指标的输出（详细版）"""
        name = metric["name"]
        value = metric["value"]
        unit = metric["unit"]
        ref = metric["reference_range"] or "暂无参考范围"
        category = metric["category"]

        health_interpretation = metric["health_interpretation"]
        suggestion = metric["suggestion"]
        applicable = metric["applicable"]
        professional_interpretation = metric["professional_interpretation"]

        lines = []
        lines.append(f"{name}为{value}{unit}，参考范围：{ref}，属于 {category} 类指标。")
        # if health_interpretation:
        #     lines.append(f"健康解读：{health_interpretation}")
        # if suggestion:
        #     lines.append(f"诊疗建议：{suggestion}")
        # if applicable:
        #     lines.append(f"适用范围：{applicable}")
        # if professional_interpretation:
        #     lines.append(f"专业解读：{professional_interpretation}")
        
        return "\n".join(lines)
    
    # 分离正常指标和异常指标
    normal_metrics = []
    abnormal_metrics = []
    
    for metric in data["analysis"]:
        value = metric["value"]
        reference_range = metric["reference_range"]
        
        if is_abnormal(value, reference_range):
            abnormal_metrics.append(metric)
        else:
            normal_metrics.append(metric)
    
    # 生成报告内容
    report_lines = []
    
    # 先输出正常指标（简洁版，只显示基本信息）
    if normal_metrics:
        report_lines.append("【正常指标】")
        report_lines.append("")
        
        for metric in normal_metrics:
            report_lines.append(format_metric_simple(metric))
            report_lines.append("")  # 空行分隔
    
    # 再输出异常指标（详细版，包含解读和建议）
    if abnormal_metrics:
        report_lines.append("【异常指标】")
        report_lines.append("")
        
        for metric in abnormal_metrics:
            report_lines.append(format_metric_detailed(metric))
            report_lines.append("")  # 空行分隔
    
    return "\n".join(report_lines)

def interpret_abnormal_metrics(report: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> str:
    """
    解读异常指标
    
    Args:
        report: 报告信息
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    """
    prompt_template = get_prompt('ABNORMAL_METRIC_INTERPRETATION_PROMPT', language)
    prompt = prompt_template.format(report=report)
    try:
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
                client=client,
                model=model,
                use_mcp=False,
                use_rag=False,
                deep_think=True
            )
        return result
    except Exception as e:
        print(f"异常指标解读与建议错误: {str(e)}")
        return "异常指标解读与建议失败"
def suggest_additional_tests(report: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> str:
    """
    建议补充检查项目
    
    Args:
        report: 分析文本
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    """
    prompt_template = get_prompt('CHECKUP_FOLLOWUP_RECOMMENDATION_PROMPT', language)
    prompt = prompt_template.format(analysis_text=report)
    try:
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_mcp=False,
            use_rag=False,
            deep_think=True
        )
        return result
    except Exception as e:
        print(f"检查项目建议错误: {str(e)}")
        return "检查项目建议失败"
def recommend_departments(report: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> str:
    """
    推荐就诊科室
    
    Args:
        report: 报告信息
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    """
    prompt_template = get_prompt('MAJOR_ABNORMAL_REFERRAL_PROMPT', language)
    prompt = prompt_template.format(report=report)
    try:
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_mcp=False,
            use_rag=False,
            deep_think=True
        )
        return result
    except Exception as e:
        print(f"科室推荐错误: {str(e)}")
        return "科室推荐失败"
def summarize_to_user(original_dialogue:str,report: str, interpretations: str, checkup_suggestions: str, department_recommendations: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> str:
    """
    为用户生成总结
    
    Args:
        original_dialogue: 原始对话
        report: 报告内容
        interpretations: 解读内容
        checkup_suggestions: 检查建议
        department_recommendations: 科室推荐
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b
    """
    prompt_template = get_prompt('SUMMARIZE_TO_USER_PROMPT', language)
    prompt = prompt_template.format(dialogue=original_dialogue, report=report, interpretations=interpretations, checkup_suggestions=checkup_suggestions, department_recommendations=department_recommendations)
    try:
        result = chat_with_llm(
            messages=[{"role": "user", "content": prompt}],
            client=client,
            model=model,
            use_mcp=False,
            use_rag=False,
            deep_think=True
        )
        return result
    except Exception as e:
        print(f"总结到用户错误: {str(e)}")
        return "总结到用户失败"
def metrics_to_natural_language(data: dict, language: str = 'zh') -> str:
    metrics = data.get("metrics", [])
    confidence = data.get("extraction_confidence", "未知" if language == 'zh' else "Unknown")
    missing_info = data.get("missing_info", "")

    lines = []

    if not metrics:
        return "未提取到有效的体检指标数据。" if language == 'zh' else "No valid health metrics were extracted."

    # 翻译置信度值
    confidence_map = {
        'zh': {'高': '高', '中': '中', '低': '低', 'High': '高', 'Medium': '中', 'Low': '低'},
        'en': {'高': 'High', '中': 'Medium', '低': 'Low', 'High': 'High', 'Medium': 'Medium', 'Low': 'Low'}
    }
    confidence_display = confidence_map.get(language, {}).get(confidence, confidence)
    
    # 根据语言选择文本
    if language == 'zh':
        lines.append(f"以下是系统提取到的体检指标信息，数据提取可信度为\"{confidence_display}\"：")
    else:
        lines.append(f"The following is the health metrics information extracted by the system, with extraction confidence: \"{confidence_display}\":")

    for metric in metrics:
        name = metric.get("name", "未知指标" if language == 'zh' else "Unknown metric")
        value = metric.get("value", "未知" if language == 'zh' else "Unknown")
        unit = metric.get("unit", "")
        category = metric.get("category", "其他" if language == 'zh' else "Other")

        if language == 'zh':
            line = f"- {name}：为 {value}{unit}，属于{category}类指标。"
        else:
            line = f"- {name}: {value}{unit}, belongs to {category} category."
        lines.append(line)

    if missing_info and ("未发现" not in missing_info and "not found" not in missing_info.lower()):
        if language == 'zh':
            lines.append(f"\n⚠️ 补充说明：{missing_info}")
        else:
            lines.append(f"\n⚠️ Additional note: {missing_info}")

    return "\n".join(lines)


def _unwrap_llm_result(result: Union[str, Dict[str, Any]]) -> str:
    """将统一对话函数的返回结果规范化为字符串，并清理think/markdown围栏。"""
    try:
        if isinstance(result, dict):
            result_text = result.get("llm_response", "") or result.get("answer", "") or ""
        else:
            result_text = result
        result_text = clean_think(result_text)
        # 去除所有```标记
        import re as _re
        result_text = _re.sub(r"```[a-zA-Z]*\s*", "", result_text)
        result_text = _re.sub(r"\s*```", "", result_text)
        return result_text.strip()
    except Exception:
        return ""


def report_workflow_stream(dialogue: str, client: Client, language: str = 'zh', model: str = "qwen3:32b") -> Generator[Dict[str, Any], None, None]:
    """
    流式体检报告分析工作流：每个阶段完成后产出一条事件，用于逐步告知前端。

    Args:
        dialogue: 用户与医生的对话内容
        client: Ollama客户端实例
        language: 语言代码，'zh' 或 'en'，默认为 'zh'
        model: 使用的模型名称，默认为 qwen3:32b

    Yields 字典事件：
    - {"type": "content", "content": str}  用于直接流式输出到前端主内容区
    - {"type": "stage", "stage": str, "message": str} 可选，用于阶段性状态
    最后会再产出一条 {"type": "content", "content": 最终总结}。
    """
    # 创建工作流日志记录器
    logger = WorkflowLogger()
    
    try:
        # 记录输入对话
        logger.log_stage("输入对话", dialogue)
        
        # Step 1: 指标抽取
        extracted_metrics = extract_metrics_from_dialogue(dialogue, client, language, model=model)
        metrics_count = extracted_metrics.get("metrics_count", len(extracted_metrics.get("metrics", [])))
        logger.log_stage("Step1_指标提取", extracted_metrics)
        print(f"指标提取完成，共提取到 {metrics_count} 个指标")
        
        extracted_metrics_nl = metrics_to_natural_language(extracted_metrics, language)
        logger.log_stage("Step1_指标自然语言", extracted_metrics_nl)
        
        title = "阶段1：指标提取完成" if language == 'zh' else "Stage 1: Metrics extracted"
        metrics_count_text = f"（共提取到 {metrics_count} 个指标）" if language == 'zh' else f"(Extracted {metrics_count} metrics in total)"
        yield {
            "type": "stage",
            "stage": 1,
            "title": title,
            "content": f"{metrics_count_text}\n\n{extracted_metrics_nl}"
        }

        if not extracted_metrics or not extracted_metrics.get("metrics"):
            error_msg = "未提取到有效的体检指标数据，工作流结束。" if language == 'zh' else "No valid health metrics were extracted. Workflow terminated."
            logger.log_stage("Step1_错误", error_msg)
            logger.finalize()
            title = "阶段1：指标提取失败" if language == 'zh' else "Stage 1: Metrics extraction failed"
            yield {
                "type": "stage",
                "stage": 1,
                "title": title,
                "content": error_msg
            }
            return

        # Step 2: 载入参考指标数据
        # 根据语言选择不同的参考数据文件
        csv_path = "app/util/tijian_en.csv" if language == 'en' else "app/util/tijian.csv"
        reference_data = load_reference_metrics(csv_path, language)
        logger.log_stage("Step2_参考数据加载", reference_data)
        
        if not reference_data:
            error_msg = "未能加载参考指标数据，工作流结束。" if language == 'zh' else "Failed to load reference metrics data. Workflow terminated."
            logger.log_stage("Step2_错误", error_msg)
            logger.finalize()
            yield {
                "type": "content",
                "content": f"{error_msg}\n"
            }
            return
            
        title = "阶段2：参考指标数据加载完成" if language == 'zh' else "Stage 2: Reference metrics loaded"
        yield {
            "type": "stage",
            "stage": 2,
            "title": title,
            "content": ""
        }

        # Step 3: 指标异常分析
        analyzed_results = analyze_uploaded_metrics(extracted_metrics, reference_data, language)
        logger.log_stage("Step3_指标分析", analyzed_results)
        
        if not analyzed_results or not analyzed_results.get("analysis"):
            error_msg = "指标分析失败，工作流结束。" if language == 'zh' else "Metrics analysis failed. Workflow terminated."
            logger.log_stage("Step3_错误", error_msg)
            logger.finalize()
            title = "阶段3：指标分析失败" if language == 'zh' else "Stage 3: Metrics analysis failed"
            yield {
                "type": "stage",
                "stage": 3,
                "title": title,
                "content": error_msg
            }
            return
        
        # 生成报告但不在前端显示，只用于后续分析
        report_text = generate_report(analyzed_results)
        logger.log_stage("Step3_体检报告生成", report_text)
        
        # 分类收集正常和异常指标
        normal_metrics = []
        abnormal_metrics = []
        
        for metric in analyzed_results.get("analysis", []):
            value = metric.get("value", "")
            reference_range = metric.get("reference_range", "")
            
            # 判断是否异常
            is_abnormal = False
            
            # 将value转换为字符串以便检查
            value_str = str(value).strip().lower()
            
            # 检查是否为"阳性"（包括各种可能的表示方式）
            positive_keywords = ["阳性", "positive", "+", "pos", "异常", "abnormal"]
            if any(keyword in value_str for keyword in positive_keywords):
                is_abnormal = True
            else:
                try:
                    no_ref_text_zh = "暂无参考范围"
                    no_ref_text_en = "No reference range available"
                    if reference_range and reference_range != no_ref_text_zh and reference_range != no_ref_text_en:
                        metric_value = float(value)
                        range_str = reference_range.split()[0] if ' ' in reference_range else reference_range
                        
                        if '-' in range_str:
                            parts = range_str.split('-')
                            if len(parts) == 2:
                                lower = float(parts[0])
                                upper = float(parts[1])
                                if metric_value < lower or metric_value > upper:
                                    is_abnormal = True
                except:
                    pass
            
            # 分类存储
            if is_abnormal:
                abnormal_metrics.append(metric)
            else:
                normal_metrics.append(metric)
        
        # 构建显示内容
        if language == 'zh':
            content_lines = ["指标分析完成"]
            
            # 正常指标（显示前3个）
            if normal_metrics:
                content_lines.append(f"✅ **正常指标**（共 {len(normal_metrics)} 个）：\n")
                for metric in normal_metrics[:3]:
                    name = metric.get("name", "未知")
                    value = metric.get("value", "")
                    unit = metric.get("unit", "")
                    reference_range = metric.get("reference_range", "")
                    content_lines.append(f"- {name}：{value}{unit}（参考范围：{reference_range}）")
                if len(normal_metrics) > 3:
                    content_lines.append(f"- ... 还有 {len(normal_metrics) - 3} 个正常指标")
                content_lines.append("")
            
            # 异常指标（显示前3个）
            if abnormal_metrics:
                content_lines.append(f"⚠️ **异常指标**（共 {len(abnormal_metrics)} 个）：\n")
                for metric in abnormal_metrics:
                    name = metric.get("name", "未知")
                    value = metric.get("value", "")
                    unit = metric.get("unit", "")
                    reference_range = metric.get("reference_range", "")
                    content_lines.append(f"- {name}：{value}{unit}（参考范围：{reference_range}）")
                content_lines.append("")
        else:
            content_lines = ["Metrics analysis completed"]
            
            # 正常指标（显示前3个）
            if normal_metrics:
                content_lines.append(f"✅ **Normal Metrics** ({len(normal_metrics)} total):\n")
                for metric in normal_metrics[:3]:
                    name = metric.get("name", "Unknown")
                    value = metric.get("value", "")
                    unit = metric.get("unit", "")
                    reference_range = translate_reference_range(metric.get("reference_range", ""), language)
                    content_lines.append(f"- {name}: {value}{unit} (Reference range: {reference_range})")
                if len(normal_metrics) > 3:
                    content_lines.append(f"- ... {len(normal_metrics) - 3} more normal metrics")
                content_lines.append("")
            
            # 异常指标（显示前3个）
            if abnormal_metrics:
                content_lines.append(f"⚠️ **Abnormal Metrics** ({len(abnormal_metrics)} total):\n")
                for metric in abnormal_metrics:
                    name = metric.get("name", "Unknown")
                    value = metric.get("value", "")
                    unit = metric.get("unit", "")
                    reference_range = translate_reference_range(metric.get("reference_range", ""), language)
                    content_lines.append(f"- {name}: {value}{unit} (Reference range: {reference_range})")
                content_lines.append("")
        
        title = "阶段3：指标分析完成" if language == 'zh' else "Stage 3: Metrics analysis completed"
        yield {
            "type": "stage",
            "stage": 3,
            "title": title,
            "content": "\n".join(content_lines)
        }

        # Step 4: 异常指标解读与建议
        interpretations = interpret_abnormal_metrics(report_text, client, language, model=model)
        logger.log_stage("Step4_异常指标解读", interpretations)
        
        interpretations_text = _unwrap_llm_result(interpretations)
        if interpretations_text:
            title = "阶段4：异常指标解读与建议" if language == 'zh' else "Stage 4: Interpretation & recommendations"
            yield {
                "type": "stage",
                "stage": 4,
                "title": title,
                "content": interpretations_text
            }

        # Step 5: 检查项目建议
        checkup_suggestions = suggest_additional_tests(report_text, client, language, model=model)
        logger.log_stage("Step5_检查项目建议", checkup_suggestions)
        
        checkup_suggestions_text = _unwrap_llm_result(checkup_suggestions)
        if checkup_suggestions_text:
            title = "阶段5：检查项目建议" if language == 'zh' else "Stage 5: Follow-up test suggestions"
            yield {
                "type": "stage",
                "stage": 5,
                "title": title,
                "content": checkup_suggestions_text
            }

        # Step 6: 科室推荐
        department_recommendations = recommend_departments(report_text, client, language, model=model)
        logger.log_stage("Step6_科室推荐", department_recommendations)
        
        department_recommendations_text = _unwrap_llm_result(department_recommendations)
        if department_recommendations_text:
            title = "阶段6：科室推荐" if language == 'zh' else "Stage 6: Department recommendations"
            yield {
                "type": "stage",
                "stage": 6,
                "title": title,
                "content": department_recommendations_text
            }

        # Step 7: 汇总输出
        final_response = summarize_to_user(
            original_dialogue=dialogue,
            report=report_text,
            interpretations=interpretations_text,
            checkup_suggestions=checkup_suggestions_text,
            department_recommendations=department_recommendations_text,
            client=client,
            language=language,
            model=model
        )
        logger.log_stage("Step7_最终汇总", final_response)
        
        final_text = _unwrap_llm_result(final_response)
        if final_text:
            title = "阶段7：总结" if language == 'zh' else "Stage 7: Summary"
            yield {
                "type": "stage",
                "stage": 7,
                "title": title,
                "content": final_text
            }
            
        # 完成工作流日志记录
        logger.finalize()
        
    except Exception as e:
        # 记录异常信息
        logger.log_stage("工作流异常", str(e))
        logger.finalize()
        error_msg = f"工作流发生错误：{str(e)}\n" if language == 'zh' else f"Workflow error: {str(e)}\n"
        yield {
            "type": "content",
            "content": error_msg
        }