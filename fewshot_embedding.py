# RFP 데이터의 임베딩 생성 및 벡터 스토어 저장 스크립트
#
# 이 스크립트는 다음과 같은 기능을 수행합니다:
# 1. TSV 파일에서 RFP 관련 데이터를 불러오기
# 2. 텍스트 데이터를 임베딩 벡터로 변환
# 3. Chroma 벡터 스토어에 데이터 저장
# 4. 데이터를 벡터 공간에서 검색할 수 있도록 인덱싱
#
# 주요 의존성:
# - LangChain
# - Chroma
# - HuggingFace Embeddings
# - Transformers

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from transformers import AutoTokenizer


# 1. TSV 파일에서 bidding_name 컬럼만 불러오기
df_left = pd.read_csv("data(sample)/1_infodb_bidding_information.tsv", sep="\t")
df_right = pd.read_csv("data(sample)/1_infodb_abstract_information.tsv", sep="\t")

df = pd.merge(left=df_left, right=df_right, how="inner", on="bidding_info_id").set_index("bidding_info_id")[["bidding_name", "summary"]].dropna()
df['compact'] = df.apply(lambda x: f"제목: {x.bidding_name}\n내용: {x.summary}", axis=1)

docs = [Document(page_content=row.compact, metadata={"bidding_info_id": row.Index}) for row in df.itertuples()]

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")

# 3. 텍스트 분할 (splitter는 core가 아닌 text_splitters에서 분리됨)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=400, chunk_overlap=16)
split_docs = text_splitter.split_documents(docs)

# 4. jinaai 임베딩 모델 로딩
embedding_model = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v3",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# 5. Chroma에 저장
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=persist_directory
)

# 6. Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

query = "기초학력 진단평가"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(i+1, doc.metadata)
    print(doc.page_content)
    print()


