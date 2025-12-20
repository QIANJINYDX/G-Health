#!/usr/bin/env python3
"""
简单测试脚本 - 测试基本功能
"""

import asyncio
from pathlib import Path

async def test_basic():
    """测试基本功能"""
    print("🧪 开始基本功能测试...")
    
    try:
        # 测试导入
        print("📦 测试导入...")
        from ollama import Client
        print("✅ ollama 导入成功")
        
        # 测试Ollama连接
        print("🔌 测试Ollama连接...")
        client = Client(host='http://127.0.0.1:11434')
        
        # 获取可用模型列表
        try:
            models = client.list()
            print(f"✅ Ollama连接成功，可用模型: {[model.name for model in models.models]}")
        except Exception as e:
            print(f"⚠️  Ollama连接成功，但获取模型列表失败: {e}")
            
        # 测试简单聊天
        print("💬 测试简单聊天...")
        try:
            response = client.chat(
                model="qwen3:0.6b",
                messages=[{"role": "user", "content": "你好，请简单回复一下"}]
            )
            print(f"✅ 聊天测试成功: {response.message.content}")
        except Exception as e:
            print(f"❌ 聊天测试失败: {e}")
            
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请安装必要的依赖: pip install ollama")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

async def test_mcp_imports():
    """测试MCP相关导入"""
    print("\n🔧 测试MCP导入...")
    
    try:
        import mcp
        print("✅ mcp 导入成功")
    except ImportError as e:
        print(f"❌ mcp 导入失败: {e}")
        print("请安装: pip install mcp")
    
    try:
        import fastmcp
        print("✅ fastmcp 导入成功")
    except ImportError as e:
        print(f"❌ fastmcp 导入失败: {e}")
        print("请安装: pip install fastmcp")

def main():
    """主函数"""
    print("🤖 Ollama MCP 客户端测试")
    print("=" * 40)
    
    # 运行测试
    asyncio.run(test_basic())
    asyncio.run(test_mcp_imports())
    
    print("\n📋 测试完成！")
    print("\n💡 如果所有测试都通过，你可以运行:")
    print("   python client.py")
    print("\n💡 或者使用启动脚本:")
    print("   python start_chat.py")

if __name__ == "__main__":
    main()
