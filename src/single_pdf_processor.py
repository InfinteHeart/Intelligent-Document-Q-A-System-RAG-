import os
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime

import logging

# 配置日志级别，减少调试信息输出
logging.basicConfig(level=logging.INFO)
# 禁用第三方库的调试日志
dashscope_logger = logging.getLogger('dashscope')
dashscope_logger.setLevel(logging.WARNING)
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.WARNING)

_log = logging.getLogger(__name__)


class SinglePDFProcessor:
    def __init__(
        self,
        temp_dir: Optional[str] = None,
        use_llm_reranking: bool = False,
        embedding_provider: str = "dashscope",
        answering_model: str = "qwen-turbo-latest",
        domain: str = "universal"
    ):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="pdf_rag_"))
        self.use_llm_reranking = use_llm_reranking
        self.embedding_provider = embedding_provider
        self.answering_model = answering_model
        self.domain = domain

        self.uploaded_documents: Dict[str, dict] = {}
        self.retriever = None
        self.processor = None
        self._initialized = False

        self._setup_directories()

    def _setup_directories(self):
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # 使用用户指定的固定目录
        self.markdown_dir = Path("data/stock_data/debug_data")
        self.chunks_dir = Path("data/stock_data/databases/chunked_reports")
        self.vector_db_dir = Path("data/stock_data/databases/vector_dbs")
        self.debug_data_dir = self.temp_dir / "debug_data"

        for dir_path in [self.markdown_dir, self.chunks_dir, self.vector_db_dir, self.debug_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _generate_document_id(self, pdf_path: Path) -> str:
        file_hash = hashlib.sha1(pdf_path.read_bytes()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{file_hash}_{timestamp}"

    def _convert_pdf_to_markdown(self, pdf_path: Path) -> Path:
        from src import pdf_mineru
        file_path = str(pdf_path)  # 传递完整文件路径
        
        _log.info(f"使用mineru处理PDF文件: {pdf_path.name}")
        
        # 获取task_id
        task_id = pdf_mineru.get_task_id(file_path)
        
        # 执行解析任务
        pdf_mineru.get_result(task_id)
        
        # 获取解析结果
        extract_dir = Path(task_id)
        md_path = extract_dir / "full.md"
        
        if not md_path.exists():
            _log.error(f"未找到markdown文件: {md_path}")
            raise RuntimeError(f"未找到markdown文件: {md_path}")
        
        # 复制到目标目录
        target_path = self.markdown_dir / f"{pdf_path.stem}.md"
        shutil.copy2(md_path, target_path)
        
        # 清理临时文件
        shutil.rmtree(extract_dir, ignore_errors=True)
        
        _log.info(f"PDF转换为Markdown成功: {target_path}")
        return target_path

    def _split_and_index(self, md_path: Path, document_id: str) -> dict:
        from src.text_splitter import TextSplitter

        splitter = TextSplitter()
        splitter.split_markdown_reports(
            all_md_dir=self.markdown_dir,
            output_dir=self.chunks_dir,
            chunk_size=30,
            chunk_overlap=5
        )

        json_path = self.chunks_dir / f"{md_path.stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"分块后的JSON文件不存在: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            document = json.load(f)

        document["metainfo"]["document_id"] = document_id
        document["metainfo"]["original_filename"] = md_path.stem

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)

        return document

    def _initialize_retriever_and_processor(self):
        if self._initialized:
            return

        from src.dynamic_retriever import DynamicHybridRetriever
        from src.questions_processing import QuestionsProcessor

        self.retriever = DynamicHybridRetriever(
            embedding_provider=self.embedding_provider
        )

        self.processor = QuestionsProcessor(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.chunks_dir,
            questions_file_path=None,
            new_challenge_pipeline=True,
            subset_path=None,
            parent_document_retrieval=False,
            llm_reranking=self.use_llm_reranking,
            llm_reranking_sample_size=20,
            top_n_retrieval=10,
            parallel_requests=1,
            api_provider=self.embedding_provider,
            answering_model=self.answering_model,
            full_context=False
        )

        self._initialized = True

    def upload_and_process(
        self,
        pdf_path: Union[str, Path],
        document_name: Optional[str] = None
    ) -> Dict:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        _log.info(f"开始处理PDF: {pdf_path.name}")

        document_id = self._generate_document_id(pdf_path)
        if document_name:
            document_id = f"{document_name}_{document_id}"

        _log.info(f"文档ID: {document_id}")

        # 1. PDF转换为Markdown
        md_path = self._convert_pdf_to_markdown(pdf_path)

        # 2. Markdown分块并转换为JSON
        document = self._split_and_index(md_path, document_id)

        # 3. 创建向量数据库
        self._create_vector_db()

        # 4. 初始化检索器和处理器
        self._initialize_retriever_and_processor()
        self.retriever.add_document(document_id, document)
        self.uploaded_documents[document_id] = document

        result = {
            "document_id": document_id,
            "filename": pdf_path.name,
            "document_name": document_name or pdf_path.stem,
            "status": "success",
            "chunks_count": len(document.get("content", {}).get("chunks", [])),
            "pages_count": len(document.get("content", {}).get("chunks", []))
        }

        _log.info(f"PDF处理完成: {result}")
        return result

    def answer_question(
        self,
        question: str,
        kind: str = "string",
        document_ids: Optional[List[str]] = None
    ) -> dict:
        if not self._initialized or self.retriever is None:
            raise RuntimeError("请先上传并处理PDF文件")

        if not self.uploaded_documents:
            raise RuntimeError("没有已上传的文档")

        if document_ids is None:
            document_ids = list(self.uploaded_documents.keys())

        _log.info(f"开始回答问题: {question[:50]}...")

        return self._answer_with_retrieval(question, kind, document_ids)

    def _answer_with_retrieval(self, question: str, kind: str, document_ids: List[str]) -> dict:
        from src.api_requests import APIProcessor

        if self.use_llm_reranking:
            retrieval_results = self.retriever.retrieve(
                query=question,
                document_ids=document_ids,
                llm_reranking_sample_size=20,
                top_n=10,
                llm_weight=0.7
            )
        else:
            retrieval_results = self.retriever.vector_retriever.retrieve(
                query=question,
                document_ids=document_ids,
                top_n=10
            )

        if not retrieval_results:
            return {
                "final_answer": "抱歉，未在文档中找到与问题相关的内容。",
                "step_by_step_analysis": "1. 检索阶段：未找到任何相关文本块",
                "reasoning_summary": "文档中未找到回答问题所需的信息",
                "relevant_pages": []
            }

        rag_context = self._format_retrieval_results(retrieval_results)

        api_processor = APIProcessor(provider=self.embedding_provider)
        answer_dict = api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=kind,
            model=self.answering_model,
            domain=self.domain
        )

        pages = answer_dict.get("relevant_pages", [])
        validated_pages = self._validate_page_references(pages, retrieval_results)
        answer_dict["relevant_pages"] = validated_pages

        return answer_dict

    def _format_retrieval_results(self, retrieval_results: List[Dict]) -> str:
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            page_number = result.get('page', 'N/A')
            text = result.get('text', '')
            source = result.get('document_id', '')

            if source:
                context_parts.append(f'Text from {source}, page {page_number}: \n"""\n{text}\n"""')
            else:
                context_parts.append(f'Text from page {page_number}: \n"""\n{text}\n"""')

        return "\n\n---\n\n".join(context_parts)

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list) -> list:
        if claimed_pages is None:
            claimed_pages = []

        retrieved_pages = [result['page'] for result in retrieval_results]

        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            _log.warning(f"移除 {len(removed_pages)} 个虚构页码引用: {removed_pages}")

        if len(validated_pages) < 2 and retrieval_results:
            existing_pages = set(validated_pages)
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    if len(validated_pages) >= 2:
                        break

        return validated_pages[:8]

    def _create_vector_db(self):
        """创建向量数据库，参考pipeline.py中的create_vector_dbs方法"""
        from src.ingestion import VectorDBIngestor

        _log.info(f"开始创建向量数据库，输入目录: {self.chunks_dir}, 输出目录: {self.vector_db_dir}")

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(self.chunks_dir, self.vector_db_dir)

        _log.info(f"向量数据库创建完成，存储在: {self.vector_db_dir}")

    def _create_vector_db(self):
        """创建向量数据库，参考pipeline.py中的create_vector_dbs方法"""
        from src.ingestion import VectorDBIngestor

        _log.info(f"开始创建向量数据库，输入目录: {self.chunks_dir}, 输出目录: {self.vector_db_dir}")

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(self.chunks_dir, self.vector_db_dir)

        _log.info(f"向量数据库创建完成，存储在: {self.vector_db_dir}")

    def get_uploaded_documents(self) -> List[Dict]:
        return [
            {
                "document_id": doc_id,
                "document_name": doc.get("metainfo", {}).get("document_name", doc.get("original_filename", "Unknown")),
                "chunks_count": len(doc.get("content", {}).get("chunks", [])),
                "pages_count": len(doc.get("content", {}).get("pages", []))
            }
            for doc_id, doc in self.uploaded_documents.items()
        ]

    def clear(self) -> None:
        self.uploaded_documents.clear()
        if self.retriever:
            self.retriever.clear()
        self._initialized = False
        _log.info("已清空所有上传的文档")

    def cleanup(self) -> None:
        self.clear()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            _log.info(f"已清理临时目录: {self.temp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
