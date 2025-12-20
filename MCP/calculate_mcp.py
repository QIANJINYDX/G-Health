# calculator_stdio.py
from fastmcp import FastMCP
import argparse
# 创建MCP服务器实例
mcp = FastMCP("Calculator")
 
@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个数字相加"""
    return a + b
 
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """从第一个数中减去第二个数"""
    return a - b
 
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """将两个数相乘"""
    return a * b
 
@mcp.tool()
def divide(a: float, b: float) -> float:
    """将第一个数除以第二个数"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MCP Time Server")
    parser.add_argument("--port", type=int, default=9000, help="Server port (default: 9000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (default: DEBUG)")
    args = parser.parse_args()
    # 使用stdio传输方式启动服务器
    mcp.run(
        transport="http",
        host=args.host,           # Bind to all interfaces
        port=args.port,                # Custom port
        log_level=args.log_level,        # Override global log level
    )