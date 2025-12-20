# calculator_stdio.py
import argparse
from fastmcp import FastMCP
from datetime import datetime

# 创建MCP服务器实例
mcp = FastMCP("Time")

@mcp.tool()
def get_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MCP Time Server")
    parser.add_argument("--port", type=int, default=9001, help="Server port (default: 9001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (default: DEBUG)")
    args = parser.parse_args()

    # 使用 http 传输方式启动服务器
    mcp.run(
        transport="http",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
