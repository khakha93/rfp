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






