#!/usr/bin/env python3
"""
集成启动脚本 - 测试并启动Ollama MCP持续对话客户端
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_banner():
    """打印欢迎横幅"""
    print("🤖" + "="*50 + "🤖")
    print("    Ollama MCP 持续对话客户端")
    print("🤖" + "="*50 + "🤖")
    print()

def check_ollama():
    """检查Ollama服务"""
    print("🔌 检查Ollama服务...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama服务正在运行")
            return True
        else:
            print("❌ Ollama服务未运行")
            return False
    except Exception as e:
        print(f"❌ Ollama检查失败: {e}")
        return False

def check_dependencies():
    """检查Python依赖"""
    print("📦 检查Python依赖...")
    dependencies = ['mcp', 'fastmcp', 'ollama', 'pydantic']
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖都已安装")
    return True

def run_tests():
    """运行系统测试"""
    print("\n🧪 运行系统测试...")
    try:
        result = subprocess.run(['python', 'test_full_system.py'], 
                              cwd=Path(__file__).parent, 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        
        if result.returncode == 0:
            print("✅ 系统测试通过")
            return True
        else:
            print("❌ 系统测试失败")
            print("错误输出:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ 系统测试超时")
        return False
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False

def start_chat():
    """启动聊天系统"""
    print("\n🚀 启动聊天系统...")
    
    # 启动MCP服务器
    print("📡 启动MCP服务器...")
    server_process = subprocess.Popen(
        ['python', 'server.py'],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待服务器启动
    time.sleep(3)
    
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print("❌ MCP服务器启动失败")
        print("错误:", stderr.decode())
        return False
    
    print("✅ MCP服务器启动成功")
    
    # 启动客户端
    print("💬 启动聊天客户端...")
    try:
        client_process = subprocess.run(
            ['python', 'client.py'],
            cwd=Path(__file__).parent
        )
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    finally:
        # 关闭服务器
        print("🔄 关闭MCP服务器...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✅ MCP服务器已关闭")
    
    return True

def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_ollama():
        print("\n❌ 请先启动Ollama服务:")
        print("   ollama serve")
        return
    
    if not check_dependencies():
        print("\n❌ 请先安装依赖")
        return
    
    # 询问是否运行测试
    print("\n💡 是否运行系统测试？(y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', '是']:
            if not run_tests():
                print("\n❌ 系统测试失败，但你可以尝试直接启动聊天系统")
                choice2 = input("是否继续启动聊天系统？(y/n): ").lower().strip()
                if choice2 not in ['y', 'yes', '是']:
                    return
        elif choice not in ['n', 'no', '否']:
            print("无效选择，跳过测试")
    except KeyboardInterrupt:
        print("\n👋 用户中断")
        return
    
    # 启动聊天系统
    try:
        start_chat()
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 启动聊天系统时出错: {e}")

if __name__ == "__main__":
    main()
