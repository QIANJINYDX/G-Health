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
from typing import Dict, Any

# 添加项目根目录到路径，以便导入 RAG 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.util.RAG import MedicalRAG, clean_think

app = Flask(__name__)

# 全局 RAG 实例
rag_instance: MedicalRAG = None

# 默认配置（会在 main 函数中更新）
DEFAULT_CONFIG = {
    "data_dir": "app/util/rag_data/md_files",
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
        
        # 确保 RAG 实例已初始化
        if rag_instance is None:
            initialize_rag(DEFAULT_CONFIG)
        
        # 执行查询
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
        default=5000,
        help='服务监听端口 (默认: 5000)'
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


if __name__ == "__main__":
    main()

