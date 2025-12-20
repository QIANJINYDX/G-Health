import threading
import asyncio
from typing import List, Dict, Any, Tuple
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# ---- 1) 单例：持久事件循环（后台线程） ----
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

# ---- 2) 两个模型 ----
planner_llm = Ollama(model="qwen3:0.6b", request_timeout=60.0)
answer_llm  = Ollama(model="qwen3:8b",   request_timeout=120.0)

SYSTEM_PROMPT_PLANNER = "You are a tool selection assistant. Decide which tools to call."
SYSTEM_PROMPT_ANSWER  = "You are a helpful assistant. Use the provided context to answer."

# ---- 3) 在同一 loop 中构建 MCP 工具 & Planner Agent ----
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

def build_answer_agent() -> FunctionAgent:
    return FunctionAgent(
        name="AnswerAgent",
        description="Generates the final natural language answer from tool results.",
        tools=[],
        llm=answer_llm,
        system_prompt=SYSTEM_PROMPT_ANSWER,
    )

# ---- 4) 在同一 loop 上执行 planner，一次请求一个 Context ----
def run_planner_with_logging_sync(agent: FunctionAgent, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
    async def _runner():
        tool_logs: List[Dict[str, Any]] = []
        ctx = Context(agent)  # 每次请求新建，避免跨 loop 复用
        handler = agent.run(user_input, ctx=ctx)

        async for ev in handler.stream_events():
            if isinstance(ev, ToolCall):
                args = (getattr(ev, "tool_args", None)
                        or getattr(ev, "tool_kwargs", None)
                        or getattr(ev, "input", None))
                print(f"→ Calling tool: {ev.tool_name} | args: {args}")
                tool_logs.append({"tool": ev.tool_name, "args": args})
            elif isinstance(ev, ToolCallResult):
                out = getattr(ev, "tool_output", None)
                print(f"✓ Tool {ev.tool_name} returned: {out}")
                if tool_logs:
                    tool_logs[-1]["output"] = out

        resp = await handler
        return str(resp), tool_logs
    return runtime.run(_runner())

# ---- 5) 在同一 loop 上执行 answer，一次请求一个 Context ----
def run_answer_sync(agent: FunctionAgent, final_prompt: str) -> str:
    async def _runner():
        ctx = Context(agent)
        handler = agent.run(final_prompt, ctx=ctx)
        # 如需逐 token，可在此消费 handler.stream_events()
        async for _ in handler.stream_events():
            pass
        resp = await handler
        return str(resp)
    return runtime.run(_runner())

# ---- 6) main ----
def main():
    planner_agent = build_planner_agent([
        "http://127.0.0.1:9001/mcp",  # time
        "http://127.0.0.1:9000/mcp",  # calc
    ])
    answer_agent  = build_answer_agent()

    try:
        while True:
            user_input = input("> ")
            if user_input.strip().lower() == "exit":
                break

            planner_result, tool_logs = run_planner_with_logging_sync(planner_agent, user_input)

            tool_summary = "\n".join(
                f"- 工具: {log['tool']} 参数: {log.get('args')} 结果: {log.get('output')}"
                for log in tool_logs
            )
            final_prompt = (
                f"用户问题: {user_input}\n\n"
                f"工具调用过程:\n{tool_summary}\n\n"
                f"Planner 的回答: {planner_result}\n"
                f"请用自然语言告诉用户最终答案。"
            )

            final_answer = run_answer_sync(answer_agent, final_prompt)
            print("Agent:", final_answer)
    finally:
        runtime.stop()  # 程序退出时再关掉 loop

if __name__ == "__main__":
    main()
