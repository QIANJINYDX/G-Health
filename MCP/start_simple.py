#!/usr/bin/env python3
"""
简单启动脚本 - 启动Ollama MCP持续对话客户端
不依赖uv，使用标准Python
"""

import subprocess
import sys
import os
from pathlib import Path

def check_ollama():
    """检查Ollama是否正在运行"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama服务正在运行")
            return True
        else:
            print("❌ Ollama服务未运行")
            return False
    except FileNotFoundError:
        print("❌ 未找到ollama命令，请确保已安装Ollama")
        return False

def start_server_simple():
    """使用标准Python启动MCP服务器"""
    print("🚀 启动MCP服务器...")
    try:
        # 使用标准Python启动服务器
        server_process = subprocess.Popen(
            ['python', 'server.py'],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待一下确保服务器启动
        import time
        time.sleep(2)
        
        if server_process.poll() is None:
            print("✅ MCP服务器启动成功")
            return server_process
        else:
            print("❌ MCP服务器启动失败")
            return None
    except Exception as e:
        print(f"❌ 启动MCP服务器时出错: {e}")
        return None

def main():
    """主函数"""
    print("🤖 Ollama MCP 持续对话客户端启动器")
    print("=" * 50)
    
    # 检查Ollama
    if not check_ollama():
        print("请先启动Ollama服务: ollama serve")
        return
    
    # 启动服务器
    server_process = start_server_simple()
    if not server_process:
        return
    
    print("\n📋 现在启动客户端...")
    print("💡 提示：在另一个终端中运行 'python client.py' 来启动客户端")
    print("💡 或者直接按回车键启动客户端")
    
    try:
        input("按回车键启动客户端，或Ctrl+C退出...")
        
        # 启动客户端
        print("🚀 启动客户端...")
        subprocess.run(['python', 'client.py'], cwd=Path(__file__).parent)
        
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    finally:
        # 关闭服务器
        if server_process:
            print("🔄 关闭MCP服务器...")
            server_process.terminate()
            server_process.wait()
            print("✅ MCP服务器已关闭")

if __name__ == "__main__":
    main()
