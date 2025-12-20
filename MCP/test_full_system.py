#!/usr/bin/env python3
"""
完整系统测试脚本 - 测试整个MCP系统
"""

import asyncio
import subprocess
import time
import signal
import sys
from pathlib import Path

class MCPTester:
    def __init__(self):
        self.server_process = None
        self.test_results = []
        
    def log_test(self, test_name, success, message=""):
        """记录测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = f"{status} {test_name}"
        if message:
            result += f": {message}"
        self.test_results.append((test_name, success, message))
        print(result)
        
    def test_ollama_connection(self):
        """测试Ollama连接"""
        print("🔌 测试Ollama连接...")
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log_test("Ollama连接", True, "服务正在运行")
                return True
            else:
                self.log_test("Ollama连接", False, "服务未响应")
                return False
        except subprocess.TimeoutExpired:
            self.log_test("Ollama连接", False, "连接超时")
            return False
        except FileNotFoundError:
            self.log_test("Ollama连接", False, "ollama命令未找到")
            return False
        except Exception as e:
            self.log_test("Ollama连接", False, str(e))
            return False
    
    def test_python_dependencies(self):
        """测试Python依赖"""
        print("📦 测试Python依赖...")
        dependencies = ['mcp', 'fastmcp', 'ollama', 'pydantic']
        
        for dep in dependencies:
            try:
                __import__(dep)
                self.log_test(f"导入 {dep}", True)
            except ImportError:
                self.log_test(f"导入 {dep}", False, f"模块 {dep} 未安装")
                return False
        return True
    
    def start_mcp_server(self):
        """启动MCP服务器"""
        print("🚀 启动MCP服务器...")
        try:
            self.server_process = subprocess.Popen(
                ['python', 'server.py'],
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待服务器启动
            time.sleep(3)
            
            if self.server_process.poll() is None:
                self.log_test("MCP服务器启动", True)
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                self.log_test("MCP服务器启动", False, f"进程退出: {stderr.decode()}")
                return False
        except Exception as e:
            self.log_test("MCP服务器启动", False, str(e))
            return False
    
    def test_mcp_client_import(self):
        """测试MCP客户端导入"""
        print("🔧 测试MCP客户端导入...")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from client import OllamaMCP
            self.log_test("MCP客户端导入", True)
            return True
        except Exception as e:
            self.log_test("MCP客户端导入", False, str(e))
            return False
    
    def test_simple_chat(self):
        """测试简单聊天"""
        print("💬 测试简单聊天...")
        try:
            from ollama import Client
            client = Client(host='http://127.0.0.1:11434')
            
            response = client.chat(
                model="qwen3:0.6b",
                messages=[{"role": "user", "content": "你好"}]
            )
            
            if response.message.content:
                self.log_test("简单聊天", True, "成功获取响应")
                return True
            else:
                self.log_test("简单聊天", False, "响应为空")
                return False
        except Exception as e:
            self.log_test("简单聊天", False, str(e))
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.server_process:
            print("🔄 关闭MCP服务器...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("✅ MCP服务器已关闭")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始完整系统测试")
        print("=" * 50)
        
        try:
            # 测试Ollama连接
            if not self.test_ollama_connection():
                print("❌ Ollama测试失败，停止测试")
                return False
            
            # 测试Python依赖
            if not self.test_python_dependencies():
                print("❌ 依赖测试失败，停止测试")
                return False
            
            # 测试MCP客户端导入
            if not self.test_mcp_client_import():
                print("❌ 客户端导入测试失败，停止测试")
                return False
            
            # 启动MCP服务器
            if not self.start_mcp_server():
                print("❌ 服务器启动测试失败，停止测试")
                return False
            
            # 测试简单聊天
            if not self.test_simple_chat():
                print("❌ 聊天测试失败")
                return False
            
            print("\n" + "=" * 50)
            print("🎉 所有测试通过！系统准备就绪。")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️  用户中断测试")
            return False
        except Exception as e:
            print(f"\n❌ 测试过程中出现未预期的错误: {e}")
            return False
        finally:
            self.cleanup()
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n📊 测试摘要:")
        print("-" * 30)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, message in self.test_results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name}")
            if message:
                print(f"   └─ {message}")
        
        print(f"\n总计: {passed}/{total} 测试通过")
        
        if passed == total:
            print("🎉 所有测试都通过了！")
            print("\n💡 现在你可以运行:")
            print("   python start_simple.py")
            print("   或者")
            print("   python client.py")
        else:
            print("⚠️  有些测试失败了，请检查上面的错误信息")

def main():
    """主函数"""
    tester = MCPTester()
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n⏹️  收到中断信号，正在清理...")
        tester.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 运行测试
        success = tester.run_all_tests()
        
        # 打印摘要
        tester.print_summary()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 测试过程中出现严重错误: {e}")
        tester.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
