from tools import pretty_print_docs

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_teddynote.retrievers import KiwiBM25Retriever


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

import logging

from numpy import base_repr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_teddynote.retrievers import (
    EnsembleRetriever,
    EnsembleMethod,
)


def get_retriever(top_k: int = 30) -> BaseRetriever:
    logging.info("Loading embedding model...")
    # 1. 임베딩 모델 로드 (저장 시 사용한 것과 동일해야 함)
    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    logging.info("Embedding model loaded successfully")

    logging.info("Loading Chroma vectorstore...")

    # 2. Chroma 로드 (저장했던 경로와 동일해야 함)
    persist_directory = "C:\\kha\\project\\rfpTest\\chroma_db"
    vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model)
    logging.info("Chroma vectorstore loaded successfully")

    vs_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    vs = vectorstore.get()
    docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(vs['documents'], vs['metadatas'])]
    bm25_retriever = KiwiBM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    cc_ensemble_retriever = EnsembleRetriever(
    retrievers=[vs_retriever, bm25_retriever], weights=[0.7, 0.3], method=EnsembleMethod.CC)
    return cc_ensemble_retriever

# Rerank (Cross Encoder Reranker)
def cross_encoders_reranker(retriever: BaseRetriever, top_k: int =10):
    logging.info("Starting document reranking...")

    # 모델 초기화
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

    # 상위 3개의 문서 선택
    compressor = CrossEncoderReranker(model=model, top_n=top_k)

    # 문서 압축 검색기 초기화
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compression_retriever


# Rerank (FlashRank reranker)
def flashrank_reranker(retriever: BaseRetriever, top_k: int =10):
    logging.info("Starting document reranking...")

    # 모델 초기화
    compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")

    # 문서 압축 검색기 초기화
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compression_retriever


if __name__ == "__main__":
    query_retrieve = "기초학력 진단평가"
    rerank_k = 10

    # 기본 검색
    base_retriever = get_retriever()
    logging.info(f"Retrieving documents for query: {query_retrieve}")
    results = base_retriever.invoke(query_retrieve)
    logging.info(f"Retrieved {len(results)} documents")

    # ReRank

    # cross_encoders_reranker
    cross_encoders_retriever = cross_encoders_reranker(base_retriever, top_k=rerank_k)
    cross_encoders_docs = cross_encoders_retriever.invoke(query_retrieve)
    
    # flashrank_reranker
    flashrank_retriever = flashrank_reranker(base_retriever, top_k=rerank_k)
    flashrank_docs = flashrank_retriever.invoke(query_retrieve)

    # 결과 비교
    # 기본 검색
    logging.info("Starting document compression...")
    pretty_print_docs(results[:rerank_k])
    
    # cross_encoders_reranker
    logging.info("cross_encoders documents:")
    pretty_print_docs(cross_encoders_docs)

    # flashrank_reranker
    logging.info("flashrank documents:")
    pretty_print_docs(flashrank_docs[:rerank_k])

    logging.info("Document compression completed successfully")