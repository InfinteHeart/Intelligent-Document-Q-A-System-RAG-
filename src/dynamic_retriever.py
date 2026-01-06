import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import faiss
import numpy as np
import hashlib
import time

from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class DynamicVectorRetriever:
    def __init__(self, embedding_provider: str = "dashscope"):
        self.embedding_provider = embedding_provider.lower()
        self.documents: Dict[str, dict] = {}
        self.vector_dbs: Dict[str, faiss.Index] = {}
        self._initialize_embedding_client()

    def _initialize_embedding_client(self):
        from dotenv import load_dotenv
        load_dotenv()
        import os
        
        if self.embedding_provider == "openai":
            from openai import OpenAI
            self.llm = OpenAI(timeout=None, max_retries=2)
        elif self.embedding_provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            self.llm = None
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _get_embedding(self, text: str) -> List[float]:
        if self.embedding_provider == "openai":
            embedding = self.llm.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return embedding.data[0].embedding
        elif self.embedding_provider == "dashscope":
            import dashscope
            rsp = dashscope.TextEmbedding.call(
                model="text-embedding-v1",
                input=[text]
            )
            if 'output' in rsp and 'embeddings' in rsp['output']:
                emb = rsp['output']['embeddings'][0]
                if emb['embedding'] is None or len(emb['embedding']) == 0:
                    raise RuntimeError(f"DashScope返回的embedding为空")
                return emb['embedding']
            else:
                raise RuntimeError(f"DashScope embedding API返回格式异常: {rsp}")

    def _create_vector_db(self, embeddings: List[List[float]]) -> faiss.Index:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        return index

    def add_document(self, document_id: str, document: dict) -> None:
        chunks = document.get("content", {}).get("chunks", [])
        if not chunks:
            _log.warning(f"文档 {document_id} 没有内容块")
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        texts = [t[:2048] for t in texts if t]

        if not texts:
            _log.warning(f"文档 {document_id} 没有有效文本内容")
            return

        embeddings = []
        for text in texts:
            emb = self._get_embedding(text)
            embeddings.append(emb)

        index = self._create_vector_db(embeddings)

        self.documents[document_id] = document
        self.vector_dbs[document_id] = index
        _log.info(f"文档 {document_id} 已添加，包含 {len(chunks)} 个分块")

    def retrieve(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        top_n: int = 5
    ) -> List[Dict]:
        if document_ids is None:
            document_ids = list(self.vector_dbs.keys())

        if not document_ids:
            raise ValueError("没有可检索的文档")

        query_embedding = self._get_embedding(query)
        query_array = np.array([query_embedding], dtype=np.float32)

        all_results = []

        for doc_id in document_ids:
            if doc_id not in self.vector_dbs:
                continue

            index = self.vector_dbs[doc_id]
            document = self.documents[doc_id]
            chunks = document.get("content", {}).get("chunks", [])

            distances, indices = index.search(query_array, min(top_n, len(chunks)))

            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(chunks):
                    chunk = chunks[idx]
                    result = {
                        "distance": round(float(distance), 4),
                        "document_id": doc_id,
                        "page": chunk.get("page", 0),
                        "text": chunk.get("text", "")
                    }
                    all_results.append(result)

        all_results.sort(key=lambda x: x["distance"], reverse=True)
        return all_results[:top_n]

    def get_all_documents(self) -> List[dict]:
        return list(self.documents.values())

    def get_document_count(self) -> int:
        return len(self.documents)

    def clear(self) -> None:
        self.documents.clear()
        self.vector_dbs.clear()


class DynamicHybridRetriever:
    def __init__(self, embedding_provider: str = "dashscope"):
        self.vector_retriever = DynamicVectorRetriever(embedding_provider)
        self.reranker = LLMReranker()

    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        llm_reranking_sample_size: int = 20,
        top_n: int = 5,
        llm_weight: float = 0.7
    ) -> List[Dict]:
        if document_ids is None:
            document_ids = list(self.vector_retriever.vector_dbs.keys())

        if not document_ids:
            return []

        vector_results = self.vector_retriever.retrieve(
            query=query,
            document_ids=document_ids,
            top_n=llm_reranking_sample_size
        )

        if not vector_results:
            return []

        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=vector_results,
            documents_batch_size=10,
            llm_weight=llm_weight
        )

        return reranked_results[:top_n]

    def add_document(self, document_id: str, document: dict) -> None:
        self.vector_retriever.add_document(document_id, document)

    def clear(self) -> None:
        self.vector_retriever.clear()
