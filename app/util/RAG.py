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
import copy
import yaml
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Callable
from dataclasses import dataclass, field

from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.schema import Document

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


def clean_think(text: str) -> str:
    """移除思考过程标签内容，避免把"思考过程"返回给用户。
    支持 <think>...</think> 和 <think>...</think> 两种格式。
    """
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ========== Markdown 结构化分块相关类 ==========
@dataclass
class Chunk:
    """用于存储文本片段及相关元数据的类。"""
    content: str = ''
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """重写 __str__ 方法，使其仅包含 content 和 metadata。"""
        if self.metadata:
            return f"content='{self.content}' metadata={self.metadata}"
        else:
            return f"content='{self.content}'"

    def __repr__(self) -> str:
        return self.__str__()


class LineType(TypedDict):
    """行类型，使用类型字典定义。"""
    metadata: Dict[str, str]  # 元数据字典
    content: str  # 行内容


class HeaderType(TypedDict):
    """标题类型，使用类型字典定义。"""
    level: int  # 标题级别
    name: str  # 标题名称 (例如, 'Header 1')
    data: str  # 标题文本内容


class MarkdownHeaderTextSplitter:
    """基于指定的标题分割 Markdown 文件，并可选地根据 chunk_size 进一步细分。"""

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ],
        strip_headers: bool = False,
        chunk_size: Optional[int] = None,
        length_function: Callable[[str], int] = len,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
    ):
        """创建一个新的 MarkdownHeaderTextSplitter。

        Args:
            headers_to_split_on: 用于分割的标题级别和名称元组列表。
            strip_headers: 是否从块内容中移除标题行。
            chunk_size: 块的最大非代码内容长度。如果设置，将进一步分割超出的块。
            length_function: 用于计算文本长度的函数。
            separators: 用于分割的分隔符列表，优先级从高到低。
            is_separator_regex: 是否将分隔符视为正则表达式。
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数或 None。")

        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        self.strip_headers = strip_headers
        self._chunk_size = chunk_size
        self._length_function = length_function
        # 设置默认分隔符，优先段落，其次换行
        self._separators = separators or [
            "\n\n",  # 段落
            "\n",    # 行
            "。|！|？",  # 中文句末标点
            "\.\s|\!\s|\?\s", # 英文句末标点加空格
            "；|;\s",  # 分号
            "，|,\s"   # 逗号
        ]
        self._is_separator_regex = is_separator_regex
        # 预编译正则表达式（如果需要）
        self._compiled_separators = None
        if self._is_separator_regex:
            self._compiled_separators = [re.compile(s) for s in self._separators]

    def _calculate_length_excluding_code(self, text: str) -> int:
        """计算文本长度，不包括代码块内容。"""
        total_length = 0
        last_end = 0
        # 正则表达式查找 ```...``` 或 ~~~...~~~ 代码块
        # 使用非贪婪匹配 .*?
        for match in re.finditer(r"(?:```|~~~).*?\n(?:.*?)(?:```|~~~)", text, re.DOTALL | re.MULTILINE):
            start, end = match.span()
            # 添加此代码块之前的文本长度
            total_length += self._length_function(text[last_end:start])
            last_end = end
        # 添加最后一个代码块之后的文本长度
        total_length += self._length_function(text[last_end:])
        return total_length

    def _find_best_split_point(self, lines: List[str]) -> int:
        """在行列表中查找最佳分割点（索引）。

        优先寻找段落分隔符（连续两个换行符），其次是单个换行符。
        从后向前查找，返回分割点 *之后* 的那一行索引。
        如果找不到合适的分隔点（例如只有一行），返回 -1。
        """
        if len(lines) <= 1:
            return -1

        # 优先查找段落分隔符 "\n\n"
        # 这对应于一个空行
        for i in range(len(lines) - 2, 0, -1):  # 从倒数第二行向前找到第二行
            if not lines[i].strip() and lines[i+1].strip():  # 当前行是空行，下一行不是
                # 检查前一行也不是空行，确保是段落间的分隔
                if i > 0 and lines[i-1].strip():
                    return i + 1  # 在空行之后分割

        # 如果没有找到段落分隔符，则在最后一个换行符处分割
        # （即在倒数第二行之后分割）
        if len(lines) > 1:
            return len(lines) - 1  # 在倒数第二行之后分割（即保留最后一行给下一个块）

        return -1  # 理论上如果行数>1总会找到换行符，但作为保险

    def _split_chunk_by_size(self, chunk: Chunk) -> List[Chunk]:
        """将超出 chunk_size 的块分割成更小的块，优先使用分隔符。"""
        if self._chunk_size is None:  # 如果未设置 chunk_size，则不分割
            return [chunk]

        sub_chunks = []
        current_lines = []
        current_non_code_len = 0
        in_code = False
        code_fence = None
        lines = chunk.content.split('\n')

        for line_idx, line in enumerate(lines):
            stripped_line = line.strip()
            is_entering_code = False
            is_exiting_code = False

            # --- 代码块边界检查 ---
            if not in_code:
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    is_entering_code = True
                    code_fence = "```"
                elif stripped_line.startswith("~~~") and stripped_line.count("~~~") == 1:
                    is_entering_code = True
                    code_fence = "~~~"
            elif in_code and code_fence is not None and stripped_line.startswith(code_fence):
                is_exiting_code = True
            # --- 代码块边界检查结束 ---

            # --- 计算行长度贡献 ---
            line_len_contribution = 0
            if not in_code and not is_entering_code:
                line_len_contribution = self._length_function(line) + 1  # +1 for newline
            elif is_exiting_code:
                line_len_contribution = self._length_function(line) + 1
            # --- 计算行长度贡献结束 ---

            # --- 检查是否需要分割 ---
            split_needed = (
                line_len_contribution > 0 and
                current_non_code_len + line_len_contribution > self._chunk_size and
                current_lines  # 必须已有内容才能分割
            )

            if split_needed:
                # 尝试找到最佳分割点
                split_line_idx = self._find_best_split_point(current_lines)

                if split_line_idx != -1 and split_line_idx > 0:  # 确保不是在第一行就分割
                    lines_to_chunk = current_lines[:split_line_idx]
                    remaining_lines = current_lines[split_line_idx:]

                    # 创建并添加上一个子块
                    content = "\n".join(lines_to_chunk)
                    sub_chunks.append(Chunk(content=content, metadata=chunk.metadata.copy()))

                    # 开始新的子块，包含剩余行和当前行
                    current_lines = remaining_lines + [line]
                    # 重新计算新 current_lines 的非代码长度
                    current_non_code_len = self._calculate_length_excluding_code("\n".join(current_lines))

                else:  # 找不到好的分割点或 current_lines 太短，执行硬分割
                    content = "\n".join(current_lines)
                    sub_chunks.append(Chunk(content=content, metadata=chunk.metadata.copy()))
                    current_lines = [line]
                    current_non_code_len = line_len_contribution if not is_entering_code else 0

            else:  # 不需要分割，将行添加到当前子块
                current_lines.append(line)
                if line_len_contribution > 0:
                    current_non_code_len += line_len_contribution
            # --- 检查是否需要分割结束 ---

            # --- 更新代码块状态 ---
            if is_entering_code:
                in_code = True
            elif is_exiting_code:
                in_code = False
                code_fence = None
            # --- 更新代码块状态结束 ---

        # 添加最后一个子块
        if current_lines:
            content = "\n".join(current_lines)
            # 最后检查一次这个块是否超长（可能只有一个元素但超长）
            final_non_code_len = self._calculate_length_excluding_code(content)
            if final_non_code_len > self._chunk_size and len(sub_chunks) > 0:
                # 如果最后一个块超长，并且不是唯一的块，可能需要警告或特殊处理
                # 这里简单地添加它，即使它超长
                pass  # logger.warning(f"Final chunk exceeds chunk_size: {final_non_code_len} > {self._chunk_size}")
            sub_chunks.append(Chunk(content=content, metadata=chunk.metadata.copy()))

        return sub_chunks if sub_chunks else [chunk]

    def _aggregate_lines_to_chunks(self, lines: List[LineType],
                                   base_meta: dict) -> List[Chunk]:
        """将具有共同元数据的行合并成块。"""
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]["metadata"] == line["metadata"]:
                # 追加内容，保留换行符
                aggregated_chunks[-1]["content"] += "\n" + line["content"]
            else:
                # 创建新的聚合块，使用 copy 防止后续修改影响
                aggregated_chunks.append(copy.deepcopy(line))

        final_chunks = []
        for chunk_data in aggregated_chunks:
            final_metadata = base_meta.copy()
            final_metadata.update(chunk_data['metadata'])
            # 在这里移除 strip()，因为后续的 _split_chunk_by_size 需要原始换行符
            final_chunks.append(
                Chunk(content=chunk_data["content"],  # 移除 .strip()
                      metadata=final_metadata)
            )
        return final_chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """基于标题分割 Markdown 文本，并根据 chunk_size 进一步细分。"""
        base_metadata = metadata or {}
        lines = text.split("\n")
        lines_with_metadata: List[LineType] = []
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        header_stack: List[HeaderType] = []

        in_code_block = False
        opening_fence = ""

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()

            # --- 代码块处理逻辑开始 ---
            # 检查是否是代码块开始或结束标记
            is_code_fence = False
            if not in_code_block:
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                    is_code_fence = True
                elif stripped_line.startswith("~~~") and stripped_line.count("~~~") == 1:
                    in_code_block = True
                    opening_fence = "~~~"
                    is_code_fence = True
            # 检查是否是匹配的结束标记
            elif in_code_block and opening_fence is not None and stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ""
                is_code_fence = True
            # --- 代码块处理逻辑结束 ---

            # 如果在代码块内（包括边界行），直接添加到当前内容
            if in_code_block or is_code_fence:
                current_content.append(line)
                continue  # 继续下一行，不检查标题

            # --- 标题处理逻辑开始 (仅在代码块外执行) ---
            found_header = False
            for sep, name in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                ):
                    found_header = True
                    header_level = sep.count("#")
                    header_data = stripped_line[len(sep):].strip()

                    # 如果找到新标题，且当前有内容，则将之前的内容聚合
                    if current_content:
                        lines_with_metadata.append({
                            "content": "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        })
                        current_content = []  # 重置内容

                    # 更新标题栈
                    while header_stack and header_stack[-1]["level"] >= header_level:
                        header_stack.pop()
                    new_header: HeaderType = {"level": header_level, "name": name, "data": header_data}
                    header_stack.append(new_header)
                    current_metadata = {h["name"]: h["data"] for h in header_stack}

                    # 如果不剥离标题，则将标题行添加到新内容的开始
                    if not self.strip_headers:
                        current_content.append(line)

                    break  # 找到匹配的最高级标题后停止检查
            # --- 标题处理逻辑结束 ---

            # 如果不是标题行且不在代码块内
            if not found_header:
                # 只有当行不为空或当前已有内容时才添加（避免添加文档开头的空行）
                # 或者保留空行以维持格式
                if line.strip() or current_content:
                    current_content.append(line)

        # 处理文档末尾剩余的内容
        if current_content:
            lines_with_metadata.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy(),
            })

        # 第一步：基于标题聚合块
        aggregated_chunks = self._aggregate_lines_to_chunks(lines_with_metadata, base_meta=base_metadata)

        # 第二步：如果设置了 chunk_size，则进一步细分块
        if self._chunk_size is None:
            return aggregated_chunks  # 如果没有 chunk_size，直接返回聚合块
        else:
            final_chunks = []
            for chunk in aggregated_chunks:
                # 检查块的非代码内容长度
                non_code_len = self._calculate_length_excluding_code(chunk.content)

                if non_code_len > self._chunk_size:
                    # 如果超出大小，则进行细分
                    split_sub_chunks = self._split_chunk_by_size(chunk)
                    final_chunks.extend(split_sub_chunks)
                else:
                    # 如果未超出大小，直接添加
                    final_chunks.append(chunk)
            return final_chunks


class MedicalRAG:
    """
    用法示例：
        rag = MedicalRAG(
            data_dir="app/util/rag_data/mini_files",
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
        data_dir: str = "app/util/rag_data/mini_files",
        persist_dir: str = "app/util/rag_data/chroma_db",
        embed_model: str = "app/util/rag_model/ritrieve_zh_v1",
        llm_model: str = "qwen3:0.6b",
        verbose: bool = True,
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.verbose = verbose

        # 优化：显式设置 batch_size 以充分利用 GPU/CPU 吞吐
        # GPU 上建议较小的批次以避免显存溢出，CPU 上建议 32~128
        # 考虑到显存限制，使用较小的批次大小
        embedding_batch_size = 32 if _DEVICE == "cuda" else 64
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={"device": _DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": embedding_batch_size
            },
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

        # 确保持久化目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # 检查向量库是否已存在且不为空
        if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
            try:
                # 检查目录是否为空
                if os.listdir(self.persist_dir):
                    _banner("检测到已有向量库，直接加载", f"persist_dir = {self.persist_dir}")
                    self.vector_store = Chroma(
                        persist_directory=self.persist_dir,
                        embedding_function=self.embeddings,
                    )
                    # 验证数据库是否真的有文档数据
                    try:
                        # 尝试获取文档数量
                        collection = self.vector_store._collection
                        # 使用get方法检查是否有数据
                        results = collection.get(limit=1)
                        doc_count = len(results.get('ids', [])) if results else 0
                        
                        # 如果数据库为空，尝试重新构建
                        if doc_count == 0:
                            if self.verbose:
                                print(f"[警告] 向量库目录存在但数据库为空（文档数：0），将重新构建")
                            # 清空目录并重新构建
                            import shutil
                            shutil.rmtree(self.persist_dir)
                            os.makedirs(self.persist_dir, exist_ok=True)
                        else:
                            # 获取实际文档总数
                            all_results = collection.get()
                            actual_count = len(all_results.get('ids', [])) if all_results else doc_count
                            # 数据库有数据，更新统计信息
                            self.stats["chunk_count"] = actual_count
                            self._init_qa(retriever_k)
                            self._summary(t0)
                            return
                    except Exception as e:
                        if self.verbose:
                            print(f"[警告] 无法验证向量库数据：{e}，将重新构建")
                        # 清空目录并重新构建
                        import shutil
                        shutil.rmtree(self.persist_dir)
                        os.makedirs(self.persist_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                # 如果无法访问目录，记录警告并继续构建
                if self.verbose:
                    print(f"[警告] 无法访问向量库目录 {self.persist_dir}：{e}，将重新构建")

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

        # 阶段 3：向量化入库（带进度，使用优化模式）
        self._ingest_chunks(chunks, ingest_batch_size, use_optimized=True)

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

    # ========== 阶段 2：切分文档（基于 Markdown 标题结构） ==========
    def _split_documents(
        self,
        documents: list,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> list:
        _banner(
            "阶段 2/4：基于 Markdown 标题结构切分文档",
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, batch_size={batch_size}",
        )
        if not documents:
            return []

        # 使用基于 Markdown 标题的分块器
        # 注意：chunk_overlap 在基于标题的分块中不直接支持，但可以通过 chunk_size 控制块的大小
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,  # 保留标题在内容中
            chunk_size=chunk_size,  # 如果块超出大小，会进一步细分
            length_function=len,
            is_separator_regex=True,  # 启用正则表达式分隔符以支持中文标点
        )

        chunks_all = []
        total = (len(documents) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(documents), batch_size), desc="切分文档批次", total=total, unit="batch"):
            batch_docs = documents[i : i + batch_size]
            for doc in batch_docs:
                # 获取文档的文本内容和元数据
                text = doc.page_content
                base_metadata = doc.metadata.copy()
                
                # 使用 MarkdownHeaderTextSplitter 分割文本
                md_chunks = splitter.split_text(text, metadata=base_metadata)
                
                # 将 Chunk 对象转换为 LangChain Document 对象
                for md_chunk in md_chunks:
                    # 合并基础元数据和标题元数据
                    final_metadata = base_metadata.copy()
                    final_metadata.update(md_chunk.metadata)
                    
                    # 创建 LangChain Document
                    langchain_doc = Document(
                        page_content=md_chunk.content,
                        metadata=final_metadata
                    )
                    chunks_all.append(langchain_doc)
        
        if self.verbose:
            print(f"切分完成：得到文本块 {len(chunks_all)} 条（基于 Markdown 标题结构）")
        return chunks_all

    # ========== 阶段 3：向量化入库（优化版：先算 embedding，再一次性 add） ==========
    def _ingest_chunks(self, chunks: list, batch_size: int, use_optimized: bool = True):
        """
        向量化并写入向量库
        
        优化策略：先批量计算所有 embedding，再一次性 add
        - 避免 Chroma 在每次 add 时重复计算 embedding
        - 减少 Chroma 内部的索引维护开销
        - 提升整体速度，避免"越跑越慢"
        
        Args:
            chunks: 文档块列表
            batch_size: 批次大小（用于 embedding 计算）
            use_optimized: 是否使用优化模式
        """
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

        import time
        start_time = time.time()
        total_chunks = len(chunks)
        
        # Chroma 的最大批次限制（安全值，实际限制约为 5461）
        CHROMA_MAX_BATCH = 5000
        
        # 步骤 1：先批量计算所有 embedding
        if self.verbose:
            print("步骤 1/2：批量计算所有 embedding...")
        
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 使用较小的批次大小以避免显存溢出
        # 根据设备类型动态调整：GPU 使用更小的批次，CPU 可以使用稍大的批次
        if _DEVICE == "cuda":
            # GPU 上使用较小的批次以避免显存溢出
            embedding_batch_size = min(32, batch_size // 4)  # 默认使用 32 或更小的批次
        else:
            # CPU 上可以使用稍大的批次
            embedding_batch_size = min(64, batch_size // 2)
        
        # 确保批次大小至少为 1
        embedding_batch_size = max(1, embedding_batch_size)
        
        if self.verbose:
            print(f"Embedding 批次大小: {embedding_batch_size} (根据显存情况自动调整)")
        
        all_embeddings = []
        
        # 尝试导入 torch 以进行显存清理
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
        
        pbar_embed = tqdm(total=total_chunks, desc="计算 embedding", unit="chunk")
        i = 0
        while i < total_chunks:
            try:
                batch_texts = texts[i : i + embedding_batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                pbar_embed.update(len(batch_texts))
                i += embedding_batch_size
                
                # 定期清理 CUDA 缓存以释放显存（每处理 10 个批次清理一次）
                if has_torch and _DEVICE == "cuda" and i % (embedding_batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and has_torch and _DEVICE == "cuda":
                    # 如果显存不足，清理缓存并减小批次大小
                    torch.cuda.empty_cache()
                    old_batch_size = embedding_batch_size
                    embedding_batch_size = max(1, embedding_batch_size // 2)
                    if self.verbose:
                        print(f"\n[警告] 显存不足，减小批次大小从 {old_batch_size} 至 {embedding_batch_size}")
                    # 不改变 i，重试当前批次（使用更小的批次）
                    if embedding_batch_size == 1:
                        # 如果批次大小已经是 1，说明单个文本块太大，尝试处理单个文本
                        if i < total_chunks:
                            try:
                                single_text = texts[i]
                                single_embedding = self.embeddings.embed_documents([single_text])
                                all_embeddings.extend(single_embedding)
                                pbar_embed.update(1)
                                i += 1
                                torch.cuda.empty_cache()
                            except RuntimeError:
                                raise RuntimeError(f"单个文本块太大，无法在 GPU 上处理。请减小 chunk_size 或使用 CPU。")
                    continue
                else:
                    raise
        
        pbar_embed.close()
        
        # 最终清理显存
        if has_torch and _DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        if self.verbose:
            elapsed_embed = time.time() - start_time
            print(f"Embedding 计算完成，用时 {elapsed_embed:.2f}s，平均速度 {total_chunks/elapsed_embed:.1f} chunks/s")
        
        # 步骤 2：使用预计算的 embedding 批量写入 Chroma
        if self.verbose:
            print("步骤 2/2：使用预计算的 embedding 批量写入 Chroma...")
        
        write_batch_size = min(batch_size * 5, CHROMA_MAX_BATCH) if use_optimized else min(batch_size, CHROMA_MAX_BATCH)
        
        if self.verbose:
            print(f"写入批次大小: {write_batch_size}")
        
        # 创建或获取 Chroma collection
        # 使用预计算的 embedding 创建 vector store
        if self.vector_store is None:
            # 首次创建：直接使用第一批数据和预计算的 embedding 创建
            first_batch_size = min(write_batch_size, total_chunks)
            first_batch_texts = texts[:first_batch_size]
            first_batch_embeddings = all_embeddings[:first_batch_size]
            first_batch_metadatas = metadatas[:first_batch_size]
            
            # 创建 vector store（使用预计算的 embedding）
            # 注意：Chroma.from_texts 不支持直接传入 embeddings，所以我们需要先创建再添加
            # 但我们可以使用底层 API 或者先创建空的再添加
            # 为了简化，我们先用第一批创建，然后用预计算的 embedding 覆盖（如果需要）
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            
            # 使用预计算的 embedding 添加第一批
            self.vector_store.add_texts(
                texts=first_batch_texts,
                embeddings=first_batch_embeddings,
                metadatas=first_batch_metadatas,
            )
            start_idx = first_batch_size
        else:
            start_idx = 0
        
        # 批量写入数据（使用预计算的 embedding）
        pbar_write = tqdm(total=total_chunks - start_idx, desc="写入 Chroma", unit="chunk", initial=start_idx)
        
        i = start_idx
        while i < total_chunks:
            batch_texts = texts[i : i + write_batch_size]
            batch_embeddings = all_embeddings[i : i + write_batch_size]
            batch_metadatas = metadatas[i : i + write_batch_size]
            batch_len = len(batch_texts)
            
            # 使用预计算的 embedding 添加
            self.vector_store.add_texts(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            
            self.stats["batch_count"] += 1
            i += batch_len
            
            # 更新进度条
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = i / elapsed
                eta = (total_chunks - i) / speed if speed > 0 else 0
                pbar_write.update(batch_len)
                pbar_write.set_postfix({
                    "chunks_in": batch_len,
                    "total": i,
                    "progress": f"{i / total_chunks * 100:.1f}%",
                    "speed": f"{speed:.1f}/s",
                    "eta": f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                })
            else:
                pbar_write.update(batch_len)
                pbar_write.set_postfix({
                    "chunks_in": batch_len,
                    "total": i,
                    "progress": f"{i / total_chunks * 100:.1f}%"
                })
        
        pbar_write.close()
        
        if self.verbose:
            total_elapsed = time.time() - start_time
            print(f"入库完成，总用时 {total_elapsed:.2f}s，平均速度 {total_chunks/total_elapsed:.1f} chunks/s")

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

        # 先检索以便返回引用（增加检索数量以过滤标题块）
        retrieve_k = max(k * 3, 10)  # 至少检索10个，或k的3倍
        retriever = self.vector_store.as_retriever(search_kwargs={"k": retrieve_k})
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

        _banner("增量入库：基于 Markdown 标题结构切分新文档", "chunk_size=500")
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,  # 保留标题在内容中
            chunk_size=500,  # 如果块超出大小，会进一步细分
            length_function=len,
            is_separator_regex=True,  # 启用正则表达式分隔符以支持中文标点
        )
        
        # 将文档转换为基于标题的块
        chunks = []
        for doc in docs:
            text = doc.page_content
            base_metadata = doc.metadata.copy()
            
            # 使用 MarkdownHeaderTextSplitter 分割文本
            md_chunks = splitter.split_text(text, metadata=base_metadata)
            
            # 将 Chunk 对象转换为 LangChain Document 对象
            for md_chunk in md_chunks:
                # 合并基础元数据和标题元数据
                final_metadata = base_metadata.copy()
                final_metadata.update(md_chunk.metadata)
                
                # 创建 LangChain Document
                langchain_doc = Document(
                    page_content=md_chunk.content,
                    metadata=final_metadata
                )
                chunks.append(langchain_doc)
        
        print(f"新文档切分得到 {len(chunks)} 个文本块（基于 Markdown 标题结构）。")

        _banner("增量入库：写入向量库", f"batch_size={batch_size}")
        # 优化：使用更大的批次减少 I/O 操作，但不超过 Chroma 的最大批次限制
        CHROMA_MAX_BATCH = 5000  # Chroma 的实际限制约为 5461，使用 5000 作为安全值
        optimized_batch_size = min(batch_size * 3, CHROMA_MAX_BATCH)
        
        # 使用 while 循环，确保不跳过任何 chunk
        total_chunks = len(chunks)
        pbar = tqdm(total=total_chunks, desc="入库批次", unit="chunk")
        
        i = 0
        while i < total_chunks:
            batch = chunks[i : i + optimized_batch_size]
            batch_len = len(batch)
            self.vector_store.add_documents(batch)
            i += batch_len  # 用实际批次长度推进，绝不漏数据
            pbar.update(batch_len)
        
        pbar.close()


# ========== 全局 RAG 实例初始化 ==========
# 控制是否启用 RAG 的全局变量
all_use_rag = False

# RAG 实例（延迟初始化）
medical_rag: Optional[MedicalRAG] = None


def get_medical_rag() -> Optional[MedicalRAG]:
    """获取全局 RAG 实例，如果未初始化则返回 None"""
    return medical_rag


def init_medical_rag(
    data_dir: str = "app/util/rag_data/mini_files",
    persist_dir: str = "app/util/rag_data/chroma_db",
    embed_model: str = "app/util/rag_model/ritrieve_zh_v1",
    llm_model: str = "qwen3:0.6b",
    verbose: bool = True,
) -> MedicalRAG:
    """
    初始化全局 RAG 实例
    
    Args:
        data_dir: 数据目录路径
        persist_dir: 持久化目录路径
        embed_model: 嵌入模型名称
        llm_model: LLM 模型名称
        verbose: 是否显示详细信息
        
    Returns:
        初始化后的 MedicalRAG 实例
    """
    global medical_rag
    if medical_rag is None:
        # 在初始化时打印设备信息
        _banner("参与构建的设备信息", f"device = {_DEVICE}")
        medical_rag = MedicalRAG(
            data_dir=data_dir,
            persist_dir=persist_dir,
            embed_model=embed_model,
            llm_model=llm_model,
            verbose=verbose,
        )
        try:
            medical_rag.build_or_load()
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            raise
    return medical_rag


# 如果 all_use_rag 为 True，自动初始化
if all_use_rag:
    try:
        medical_rag = MedicalRAG()
        medical_rag.build_or_load()
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")


# ========== 示例运行 ==========
if __name__ == "__main__":
    rag = MedicalRAG(
        data_dir="app/util/rag_data/test_files",
        persist_dir="app/util/rag_data/chroma_db",
        embed_model="app/util/rag_model/Qwen3-Embedding-0.6B",
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

