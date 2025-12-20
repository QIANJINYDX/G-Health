import subprocess
import os
import sys
from pathlib import Path

# MCP 服务启动命令列表
# 每个元素是一个完整的命令（list 或字符串）
MCP_SERVICES = [
    ["python", "time_mcp.py", "--port", "9001"],
    ["python", "calculate_mcp.py", "--port", "9000"],
    ["python", "baidumap.py", "--port", "9002"],
    ["uv", "run", "--with", "nexonco-mcp", "nexonco", "--transport", "sse", "--port", "9007"],
    ['node', 'howtocook-mcp/build/index.js','--transport','sse','--port','9008']

    # 也可以是 shell 命令，例如:
    # "node server.js"
    # "fastmcp run my_service"
]

def start_services():
    processes = []
    base_dir = Path(__file__).parent

    for cmd in MCP_SERVICES:
        # 如果传的是字符串，就用 shell=True
        if isinstance(cmd, str):
            display_cmd = cmd
            p = subprocess.Popen(cmd, cwd=base_dir, shell=True)
        else:
            display_cmd = " ".join(cmd)
            p = subprocess.Popen(cmd, cwd=base_dir)

        print(f"🚀 启动 MCP 服务: {display_cmd}")
        processes.append((display_cmd, p))

    print("\n✅ 所有 MCP 服务已启动！按 Ctrl+C 结束全部。")

    try:
        for name, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\n⏹ 检测到 Ctrl+C，正在关闭 MCP 服务...")
        for name, proc in processes:
            proc.terminate()
        print("✅ 全部 MCP 服务已关闭。")

if __name__ == "__main__":
    start_services()
