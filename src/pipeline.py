# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
import sys
from dataclasses import dataclass
from pathlib import Path
import os
import json
import pandas as pd
import shutil
import time
 
from src import pdf_mineru
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.questions_processing import QuestionsProcessor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1 # 并行的数量，需要限制，否则qwen-turbo会超出阈值
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "dashscope" 
    answering_model: str = "qwen-turbo-latest" 
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        # 此方法已不再需要，因为我们不再使用subset.csv
        pass

    def export_reports_to_markdown(self, file_names=None, batch_mode=False):
        """
        使用 pdf_mineru.py，将指定 PDF 文件或目录下所有 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        :param file_names: PDF 文件名列表（如 ['【财报】中芯国际：中芯国际2024年年度报告.pdf']）
        :param batch_mode: 是否批量处理目录下所有 PDF 文件
        """
        # 如果 batch_mode 为 True，则获取目录下所有 PDF 文件
        if batch_mode:
            input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
            file_names = [path.name for path in input_doc_paths]
        elif file_names is None:
            # 如果没有提供文件名且不是批量模式，则返回
            print("请提供文件名或启用批量处理模式")
            return
        elif isinstance(file_names, str):
            # 如果只提供了单个文件名，则转换为列表
            file_names = [file_names]
        
        # 遍历所有文件名，逐个处理
        for file_name in file_names:
            # 调用 pdf_mineru 获取 task_id 并下载、解压
            print(f"开始处理: {file_name}")
            task_id = pdf_mineru.get_task_id(file_name)
            if not task_id:
                print(f"无法获取 {file_name} 的 task_id，跳过该文件")
                continue
            print(f"task_id: {task_id}")
            pdf_mineru.get_result(task_id)

            # 解压后目录名与 task_id 相同
            extract_dir = f"{task_id}"
            md_path = os.path.join(extract_dir, "full.md")
            if not os.path.exists(md_path):
                print(f"未找到 markdown 文件: {md_path}")
                continue
            # 目标目录
            os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
            # 目标文件名为原始 file_name，扩展名改为 .md
            base_name = os.path.splitext(file_name)[0]
            target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
            shutil.move(md_path, target_path)
            print(f"已将 {md_path} 移动到 {target_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 分割markdown文件
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    # 回答用户输入问题调用这个函数
    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,
            new_challenge_pipeline=True,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        return answer

class SinglePDFPipeline:
    def __init__(
        self,
        temp_dir: str = None,
        use_llm_reranking: bool = False,
        embedding_provider: str = "dashscope",
        answering_model: str = "qwen-turbo-latest",
        domain: str = "universal"
    ):
        from src.single_pdf_processor import SinglePDFProcessor
        self.domain = domain
        self.processor = SinglePDFProcessor(
            temp_dir=temp_dir,
            use_llm_reranking=use_llm_reranking,
            embedding_provider=embedding_provider,
            answering_model=answering_model,
            domain=domain
        )

    def upload_pdf(self, pdf_path: str, document_name: str = None) -> dict:
        return self.processor.upload_and_process(pdf_path, document_name)

    def answer_question(self, question: str, kind: str = "string") -> dict:
        return self.processor.answer_question(question, kind)

    def get_documents(self) -> list:
        return self.processor.get_uploaded_documents()

    def clear(self):
        self.processor.clear()

    def cleanup(self):
        self.processor.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()