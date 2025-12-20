# -*- coding: utf-8 -*-
"""
Medical RAG（带构建说明与进度条）
- 支持 .md / .pdf / .txt 文档
- HuggingFaceEmbeddings + Chroma（持久化索引）
- 本地 Ollama 模型进行检索增强问答（RetrievalQA）
- 全流程中文说明 + tqdm 进度条

依赖（示例）：
pip install langchain langchain-community chromadb tqdm pypdf sentence-transformers

说明：
1) 首次运行会在 persist_dir 下创建并持久化索引，后续自动复用（秒级启动）。
2) 构建过程包含 4 个阶段：扫描/加载 → 切分 → 向量化入库 → 初始化 QA 链。
3) 全程显示统计信息与进度条；出错文件会跳过且记录。
"""

from __future__ import annotations
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# 为了更干净的日志（可按需注释）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 设备探测（仅用于 embedding 加速）
try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"


def _banner(title: str, sub: str = "") -> None:
    line = "━" * 28
    print(f"\n┏{line}\n┃ {title}")
    if sub:
        print(f"┃ {sub}")
    print(f"┗{line}\n")

_banner("参与构建的设备信息",f"device = {_DEVICE}")


def clean_think(text: str) -> str:
    """移除<think>…</think>内容，避免把“思考过程”返回给用户。"""
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


class MedicalRAG:
    """
    用法示例：
        rag = MedicalRAG(
            data_dir="app/util/rag_data/md_files",
            persist_dir="app/util/rag_data/chroma_db",
            embed_model="RAG/rag_model/ritrieve_zh_v1",  # 你的中文向量模型
            llm_model="qwen3:0.6b"                        # 本地 Ollama 模型
        )
        rag.build_or_load()  # 首次构建，后续直接加载
        out = rag.answer_question("糖尿病的诊断标准是什么？")
        print(out["answer"])
        print([r["source"] for r in out["references"]])
    """

    def __init__(
        self,
        data_dir: str = "app/util/rag_data/md_files",
        persist_dir: str = "app/util/rag_data/chroma_db",
        embed_model: str = "RAG/rag_model/ritrieve_zh_v1",
        llm_model: str = "qwen3:0.6b",
        verbose: bool = True,
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.verbose = verbose

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={"device": _DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None

        # 统计信息
        self.stats: Dict[str, Any] = {
            "file_total": 0,
            "file_loaded": 0,
            "file_failed": 0,
            "doc_count": 0,
            "chunk_count": 0,
            "batch_count": 0,
        }

    # ========== 核心：构建 / 加载 ==========
    def build_or_load(
        self,
        file_types: Tuple[str, ...] = (".md", ".pdf", ".txt"),
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retriever_k: int = 3,
        split_batch_size: int = 100,    # 文档切分批大小
        ingest_batch_size: int = 1000,  # 入库批大小（避免一次性嵌入太多）
    ):
        t0 = time.time()

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            _banner("检测到已有向量库，直接加载", f"persist_dir = {self.persist_dir}")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            self._init_qa(retriever_k)
            self._summary(t0)
            return

        # 阶段 1：扫描与加载
        documents = self._scan_and_load(self.data_dir, file_types=file_types)
        self.stats["doc_count"] = len(documents)

        # 阶段 2：切分文档
        chunks = self._split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=split_batch_size,
        )
        self.stats["chunk_count"] = len(chunks)

        # 阶段 3：向量化入库（带进度）
        self._ingest_chunks(chunks, ingest_batch_size)

        # 阶段 4：初始化 QA
        self._init_qa(retriever_k)

        self._summary(t0)

    # ========== 阶段 1：扫描与加载 ==========
    def _scan_and_load(self, root: str, file_types: Tuple[str, ...]) -> list:
        _banner("阶段 1/4：扫描并加载文档", f"支持后缀 = {file_types}")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"数据目录不存在：{root}")

        files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[-1].lower() in file_types
        ]
        self.stats["file_total"] = len(files)

        if self.verbose:
            print(f"发现 {len(files)} 个待处理文件；开始加载……")

        documents = []
        failed = 0
        for fp in tqdm(files, desc="加载文件", unit="file"):
            try:
                suf = os.path.splitext(fp)[-1].lower()
                if suf == ".md" or suf == ".txt":
                    documents.extend(TextLoader(fp, encoding="utf-8").load())
                elif suf == ".pdf":
                    documents.extend(PyPDFLoader(fp).load())
            except Exception as e:
                failed += 1
                if self.verbose:
                    print(f"[跳过] {os.path.basename(fp)}：{e}")

        self.stats["file_loaded"] = len(files) - failed
        self.stats["file_failed"] = failed

        if self.verbose:
            print(
                f"加载完成：成功 {self.stats['file_loaded']}，失败 {self.stats['file_failed']}，"
                f"得到原始文档 {len(documents)} 条"
            )
        return documents

    # ========== 阶段 2：切分文档 ==========
    def _split_documents(
        self,
        documents: list,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> list:
        _banner(
            "阶段 2/4：切分文档为小块",
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, batch_size={batch_size}",
        )
        if not documents:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

        chunks_all = []
        total = (len(documents) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(documents), batch_size), desc="切分文档批次", total=total, unit="batch"):
            batch_docs = documents[i : i + batch_size]
            chunks = splitter.split_documents(batch_docs)
            chunks_all.extend(chunks)
        if self.verbose:
            print(f"切分完成：得到文本块 {len(chunks_all)} 条")
        return chunks_all

    # ========== 阶段 3：向量化入库 ==========
    def _ingest_chunks(self, chunks: list, batch_size: int):
        _banner(
            "阶段 3/4：向量化并写入向量库（Chroma）",
            f"persist_dir = {self.persist_dir}, ingest_batch_size = {batch_size}",
        )
        if not chunks:
            # 仍然创建一个空库，避免后续报错
            self.vector_store = Chroma.from_documents(
                documents=[],
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )
            return

        total_batches = (len(chunks) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(chunks), batch_size), desc="入库批次", total=total_batches, unit="batch")
        for idx in pbar:
            batch = chunks[idx : idx + batch_size]
            if self.vector_store is None and idx == 0:
                # 首批：创建并持久化
                self.vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                )
            else:
                # 后续增量加入
                assert self.vector_store is not None
                self.vector_store.add_documents(batch)
            self.stats["batch_count"] += 1
            pbar.set_postfix({"chunks_in": len(batch)})

    # ========== 阶段 4：初始化 QA ==========
    def _init_qa(self, retriever_k: int):
        _banner("阶段 4/4：初始化检索问答（RetrievalQA）", f"llm_model = {self.llm_model}, top_k = {retriever_k}")
        llm = Ollama(model=self.llm_model)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": retriever_k}),
        )

    # ========== 查询接口 ==========
    def answer_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        if not self.qa_chain or not self.vector_store:
            raise RuntimeError("RAG 未初始化：请先调用 build_or_load()。")

        # 先检索以便返回引用
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        references = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source", "Unknown")
            if isinstance(src, str):
                src = os.path.basename(src)
            references.append({"content": d.page_content, "source": src})

        # LLM 生成
        result = self.qa_chain.invoke({"query": question})
        answer = clean_think(result.get("result", ""))

        if self.verbose:
            print("\n【检索到的参考文献 Top-{}】".format(k))
            for i, r in enumerate(references, 1):
                print(f"  {i}. {r['source']}")
        return {"answer": answer, "references": references}

    # ========== 构建总结 ==========
    def _summary(self, t0: float):
        dt = time.time() - t0
        _banner(
            "构建完成",
            (
                f"用时：{dt:.2f}s\n"
                f"文件：{self.stats['file_loaded']} 成功 / {self.stats['file_failed']} 失败（共 {self.stats['file_total']}）\n"
                f"原始文档数：{self.stats['doc_count']}\n"
                f"文本块（chunks）：{self.stats['chunk_count']}\n"
                f"入库批次数：{self.stats['batch_count']}\n"
                f"索引目录：{self.persist_dir}"
            ),
        )

    # ========== 可选：增量摄取新文件 ==========
    def ingest_files(self, paths: List[str], batch_size: int = 1000):
        """将新增文件增量加入现有向量库（同样带进度条）。"""
        if not self.vector_store:
            raise RuntimeError("向量库未初始化，请先 build_or_load()。")

        allowed = {".md", ".pdf", ".txt"}
        paths = [p for p in paths if os.path.splitext(p)[-1].lower() in allowed]
        if not paths:
            return

        _banner("增量入库：加载新文件", f"数量 = {len(paths)}")
        docs = []
        for p in tqdm(paths, desc="加载新文件", unit="file"):
            try:
                suf = os.path.splitext(p)[-1].lower()
                if suf in {".md", ".txt"}:
                    docs.extend(TextLoader(p, encoding="utf-8").load())
                elif suf == ".pdf":
                    docs.extend(PyPDFLoader(p).load())
            except Exception as e:
                print(f"[跳过] {os.path.basename(p)}：{e}")

        if not docs:
            print("未得到可用文档，结束。")
            return

        _banner("增量入库：切分新文档", "chunk_size=500, chunk_overlap=50")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"新文档切分得到 {len(chunks)} 个文本块。")

        _banner("增量入库：写入向量库", f"batch_size={batch_size}")
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(chunks), batch_size), desc="入库批次", total=total_batches, unit="batch"):
            self.vector_store.add_documents(chunks[i : i + batch_size])


# ========== 示例运行 ==========
if __name__ == "__main__":
    rag = MedicalRAG(
        data_dir="app/util/rag_data/md_files",
        persist_dir="app/util/rag_data/chroma_db",
        embed_model="rag_model/ritrieve_zh_v1",  # 你的中文检索模型
        llm_model="qwen3:0.6b",                       # 本地 Ollama 模型
        verbose=True,
    )

    # 首次会完整构建；之后会直接加载
    rag.build_or_load(
        file_types=(".md", ".pdf", ".txt"),
        chunk_size=500,
        chunk_overlap=50,
        retriever_k=3,
        split_batch_size=100,
        ingest_batch_size=1000,
    )

    question = "糖尿病的诊断标准是什么？"
    result = rag.answer_question(question, k=3)
    print("\n[问题]", question)
    print("\n[回答]\n", result["answer"])
    print("\n[引用来源]\n", [r["source"] for r in result["references"]])
