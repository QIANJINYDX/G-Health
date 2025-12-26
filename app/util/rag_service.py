# -*- coding: utf-8 -*-
"""
RAG Flask 服务
- 支持构建词向量数据库
- 支持启动查询服务
- 启动服务时自动检测数据库是否存在
"""

from flask import Flask, request, jsonify
import argparse
import os
import sys
import json
from typing import Dict, Any

# 添加项目根目录到路径，以便导入 RAG 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.util.RAG import MedicalRAG, clean_think

app = Flask(__name__)

# 配置 JSON 编码器，确保中文不被转义为 Unicode
app.config['JSON_AS_ASCII'] = False

# 全局 RAG 实例
rag_instance: MedicalRAG = None

# 默认配置（会在 main 函数中更新）
DEFAULT_CONFIG = {
    "data_dir": "app/util/rag_data/mini_files",
    "persist_dir": "app/util/rag_data/chroma_db",
    "embed_model": "app/util/rag_model/Qwen3-Embedding-0.6B",
    "llm_model": "qwen3:0.6b",
    "verbose": True,
}


def check_database_exists(persist_dir: str) -> bool:
    """检查数据库是否已构建"""
    if not os.path.exists(persist_dir):
        return False
    # 检查目录是否为空
    if not os.listdir(persist_dir):
        return False
    return True


def initialize_rag(config: Dict[str, Any]) -> MedicalRAG:
    """初始化 RAG 实例"""
    global rag_instance
    if rag_instance is None:
        rag_instance = MedicalRAG(
            data_dir=config["data_dir"],
            persist_dir=config["persist_dir"],
            embed_model=config["embed_model"],
            llm_model=config["llm_model"],
            verbose=config["verbose"],
        )
        # 只加载，不构建（build_or_load 会自动检测并加载已存在的数据库）
        if check_database_exists(config["persist_dir"]):
            # 调用 build_or_load，由于数据库已存在，会自动加载
            rag_instance.build_or_load()
        else:
            raise RuntimeError("数据库未构建，请先构建数据库")
    return rag_instance


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "message": "RAG service is running",
        "database_exists": check_database_exists(DEFAULT_CONFIG["persist_dir"])
    })


@app.route('/query', methods=['POST'])
def query():
    """查询接口"""
    global rag_instance
    
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body"
            }), 400
        
        question = data['question']
        k = data.get('k', 3)  # 默认返回 top 3 结果
        only_references = data.get('only_references', False)  # 是否只返回检索内容，不调用大模型
        
        # 确保 RAG 实例已初始化
        if rag_instance is None:
            initialize_rag(DEFAULT_CONFIG)
        
        if only_references:
            # 只检索，不调用大模型生成答案
            if not rag_instance.vector_store:
                raise RuntimeError("RAG 未初始化：请先调用 build_or_load()。")
            
            # 增加检索数量，以便过滤掉只有标题的块后仍有足够内容
            retrieve_k = max(k * 3, 10)  # 至少检索10个，或k的3倍
            retriever = rag_instance.vector_store.as_retriever(search_kwargs={"k": retrieve_k})
            docs = retriever.get_relevant_documents(question)
            
            # 过滤掉只有标题的块
            references = []
            seen_sources = set()  # 用于去重相同来源的块
            
            for d in docs:
                content = d.page_content.strip()
                
                # 跳过空内容
                if not content:
                    continue
                
                # 检查是否只包含标题行（以 # 开头，且内容很短或只有标题）
                lines = content.split('\n')
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                
                # 如果只有一行且是标题，跳过
                if len(non_empty_lines) == 1 and non_empty_lines[0].startswith('#'):
                    continue
                
                # 如果内容太短（少于50个字符），且主要是标题，跳过
                if len(content) < 50:
                    # 检查是否主要是标题
                    title_lines = sum(1 for line in non_empty_lines if line.startswith('#'))
                    if title_lines >= len(non_empty_lines) * 0.7:  # 70%以上是标题
                        continue
                
                meta = getattr(d, "metadata", {}) or {}
                src = meta.get("source", "Unknown")
                if isinstance(src, str):
                    src = os.path.basename(src)
                
                # 去重：如果同一个来源已经有内容，跳过（保留第一个）
                source_key = (src, content[:100])  # 使用来源和前100字符作为key
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                
                references.append({"content": content, "source": src})
                
                # 如果已经收集到足够的有效内容，停止
                if len(references) >= k:
                    break
            
            return jsonify({
                "status": "success",
                "references": references
            })
        else:
            # 执行完整查询（检索 + LLM 生成答案）
            result = rag_instance.answer_question(question, k=k)
            
            return jsonify({
                "status": "success",
                "answer": result["answer"],
                "references": result["references"]
            })
    
    except RuntimeError as e:
        return jsonify({
            "error": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "error": f"Query failed: {str(e)}"
        }), 500


@app.route('/build', methods=['POST'])
def build_database():
    """构建数据库接口"""
    global rag_instance
    
    try:
        data = request.get_json() or {}
        
        # 使用请求中的配置或默认配置
        config = {
            "data_dir": data.get("data_dir", DEFAULT_CONFIG["data_dir"]),
            "persist_dir": data.get("persist_dir", DEFAULT_CONFIG["persist_dir"]),
            "embed_model": data.get("embed_model", DEFAULT_CONFIG["embed_model"]),
            "llm_model": data.get("llm_model", DEFAULT_CONFIG["llm_model"]),
            "verbose": data.get("verbose", DEFAULT_CONFIG["verbose"]),
        }
        
        # 构建参数
        file_types = tuple(data.get("file_types", [".md", ".pdf", ".txt"]))
        chunk_size = data.get("chunk_size", 500)
        chunk_overlap = data.get("chunk_overlap", 50)
        retriever_k = data.get("retriever_k", 3)
        split_batch_size = data.get("split_batch_size", 100)
        ingest_batch_size = data.get("ingest_batch_size", 1000)
        
        # 创建新的 RAG 实例并构建
        rag = MedicalRAG(
            data_dir=config["data_dir"],
            persist_dir=config["persist_dir"],
            embed_model=config["embed_model"],
            llm_model=config["llm_model"],
            verbose=config["verbose"],
        )
        
        rag.build_or_load(
            file_types=file_types,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retriever_k=retriever_k,
            split_batch_size=split_batch_size,
            ingest_batch_size=ingest_batch_size,
        )
        
        # 更新全局实例
        rag_instance = rag
        
        return jsonify({
            "status": "success",
            "message": "Database built successfully",
            "stats": rag.stats
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Build failed: {str(e)}"
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """获取服务状态"""
    db_exists = check_database_exists(DEFAULT_CONFIG["persist_dir"])
    rag_initialized = rag_instance is not None
    
    return jsonify({
        "database_exists": db_exists,
        "rag_initialized": rag_initialized,
        "config": DEFAULT_CONFIG
    })


def build_database_cli(config: Dict[str, Any], build_params: Dict[str, Any] = None):
    """命令行构建数据库"""
    print("=" * 60)
    print("开始构建词向量数据库...")
    print("=" * 60)
    
    if build_params is None:
        build_params = {}
    
    rag = MedicalRAG(
        data_dir=config["data_dir"],
        persist_dir=config["persist_dir"],
        embed_model=config["embed_model"],
        llm_model=config["llm_model"],
        verbose=config["verbose"],
    )
    
    rag.build_or_load(
        file_types=tuple(build_params.get("file_types", [".md", ".pdf", ".txt"])),
        chunk_size=build_params.get("chunk_size", 500),
        chunk_overlap=build_params.get("chunk_overlap", 50),
        retriever_k=build_params.get("retriever_k", 3),
        split_batch_size=build_params.get("split_batch_size", 100),
        ingest_batch_size=build_params.get("ingest_batch_size", 1000),
    )
    
    print("=" * 60)
    print("数据库构建完成！")
    print("=" * 60)


def start_service(config: Dict[str, Any], host: str = "0.0.0.0", port: int = 5000):
    """启动 Flask 服务"""
    # 检查数据库是否存在
    if not check_database_exists(config["persist_dir"]):
        print("=" * 60)
        print("错误：数据库未构建！")
        print("=" * 60)
        print(f"数据库目录不存在或为空: {config['persist_dir']}")
        print("\n请先构建数据库，使用以下命令：")
        print(f"  python {__file__} --mode build")
        print("\n或者使用以下参数：")
        print(f"  python {__file__} --mode build --data-dir {config['data_dir']} --persist-dir {config['persist_dir']}")
        print("=" * 60)
        sys.exit(1)
    
    # 初始化 RAG 实例
    try:
        initialize_rag(config)
        print("=" * 60)
        print("RAG 服务启动成功！")
        print("=" * 60)
        print(f"服务地址: http://{host}:{port}")
        print(f"健康检查: http://{host}:{port}/health")
        print(f"查询接口: http://{host}:{port}/query")
        print(f"状态接口: http://{host}:{port}/status")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print("错误：初始化 RAG 实例失败！")
        print("=" * 60)
        print(f"错误信息: {str(e)}")
        print("=" * 60)
        sys.exit(1)
    
    # 启动 Flask 服务
    app.run(host=host, port=port, debug=False)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG Flask 服务')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['build', 'serve'],
        required=True,
        help='运行模式: build (构建数据库) 或 serve (启动服务)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务监听地址 (默认: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5005,
        help='服务监听端口 (默认: 5005)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_CONFIG["data_dir"],
        help=f'数据目录 (默认: {DEFAULT_CONFIG["data_dir"]})'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default=DEFAULT_CONFIG["persist_dir"],
        help=f'持久化目录 (默认: {DEFAULT_CONFIG["persist_dir"]})'
    )
    parser.add_argument(
        '--embed-model',
        type=str,
        default=DEFAULT_CONFIG["embed_model"],
        help=f'嵌入模型路径 (默认: {DEFAULT_CONFIG["embed_model"]})'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default=DEFAULT_CONFIG["llm_model"],
        help=f'LLM 模型名称 (默认: {DEFAULT_CONFIG["llm_model"]})'
    )
    
    args = parser.parse_args()
    
    config = {
        "data_dir": args.data_dir,
        "persist_dir": args.persist_dir,
        "embed_model": args.embed_model,
        "llm_model": args.llm_model,
        "verbose": True,
    }
    
    # 更新全局配置
    DEFAULT_CONFIG.update(config)
    
    if args.mode == 'build':
        build_database_cli(config)
    elif args.mode == 'serve':
        start_service(config, host=args.host, port=args.port)

# ========== HTTP 测试命令 ==========
"""
服务默认运行在 http://127.0.0.1:5005

1. 健康检查：
   curl http://127.0.0.1:5005/health

2. 查询接口（测试问题：头疼怎么治疗）：
   # 完整查询（检索 + LLM 生成答案）：
   curl -X POST http://127.0.0.1:5005/query \
     -H "Content-Type: application/json" \
     -d '{"question": "头疼怎么治疗", "k": 3}' | jq .
   
   # 只返回检索内容（不调用大模型）：
   curl -X POST http://127.0.0.1:5005/query \
     -H "Content-Type: application/json" \
     -d '{"question": "头疼怎么治疗", "k": 3, "only_references": true}' | jq .

3. 获取服务状态：
   curl http://127.0.0.1:5005/status

4. 构建数据库（如果需要）：
   curl -X POST http://127.0.0.1:5005/build \
     -H "Content-Type: application/json" \
     -d '{}'

5. 其他测试问题示例：
   # 只返回检索内容（推荐，速度快，不消耗 LLM 资源）：
   curl -X POST http://127.0.0.1:5005/query \
     -H "Content-Type: application/json" \
     -d '{"question": "糖尿病的诊断标准是什么？", "k": 3, "only_references": true}' | jq .

   curl -X POST http://127.0.0.1:5005/query \
     -H "Content-Type: application/json" \
     -d '{"question": "高血压的预防措施有哪些？", "k": 5, "only_references": true}' | jq .
   
   # 完整查询（包含 LLM 生成的答案）：
   curl -X POST http://127.0.0.1:5005/query \
     -H "Content-Type: application/json" \
     -d '{"question": "糖尿病的诊断标准是什么？", "k": 3}' | jq .
"""
if __name__ == "__main__":
    main()

