# RFP 생성을 위한 Few-Shot Learning 기반 동적 예시 선택 시스템
# 
# 이 스크립트는 다음과 같은 기능을 수행합니다:
# 1. Chroma 벡터 스토리지에서 유사한 RFP 예시를 검색
# 2. 검색된 예시들 중에서 관련도가 높은 예시들을 수동으로 재순위화
# 3. 선택된 예시들을 기반으로 Few-Shot Prompt를 생성하여 GPT-4를 통해 새로운 RFP 생성
# 4. 생성된 RFP를 마크다운 파일로 저장
#
# 주요 의존성:
# - LangChain
# - Chroma
# - HuggingFace Embeddings
# - OpenAI GPT-4

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

import pandas as pd
import os, re



# 1. 임베딩 모델 로드 (저장 시 사용한 것과 동일해야 함)
embedding_model = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v3",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Chroma 로드
persist_directory = "./chroma_db"  # 저장했던 경로와 동일해야 함
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# $$$ 이건 입력받는 걸로 바꿔야 할 듯
# 3. vectorstore -> Retrieve
query_retrieve = "기초학력 진단평가"

retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
results = retriever.invoke(query_retrieve)

bidding_info_ids = []
for i, doc in enumerate(results):
    bidding_info_ids.append(doc.metadata['bidding_info_id'])
    print(i+1, doc.metadata)
    print(doc.page_content)
    print()
# print(bidding_info_ids)


# bidding_info_id -> Bid ID
df_path = "./data(sample)/1_infodb_bidding_information.tsv"
df = pd.read_csv(df_path, sep="\t", index_col=0, usecols=[0, 4])

file_names = []
dir_path = "./fewshot/answers"
for bidding_info_id in bidding_info_ids:
    bid_no = df.loc[bidding_info_id, "bid_no"]
    file_name = os.path.join(dir_path, f"{bid_no}_refined.md")
    file_names.append(file_name)

# {questions: answers} Set 생성
questions = []
answers = []
q_suf = "의 RFP를 작성해줘"

for file_name in file_names:
    with open(file_name, "r", encoding="utf-8") as f:
        full_answer = f.read()
        answers.append(full_answer)

        title = full_answer.split("\n")[1]
        questions.append(f'"{title}"{q_suf}')
examples = [{"question": q, "answer": a} for q, a in zip(questions, answers)]


# $$$ Rerank 필요 (일단 수기로 Rerank 해봄)
rerank_examples = [examples[i] for i in [12, 15, 21, 25, 26, 29]]

# 답변 생성
example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")

# FewShotPromptTemplate을 생성합니다.
prompt = FewShotPromptTemplate(
    examples=rerank_examples,       # 사용할 예제들
    example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
    suffix="질문: {input}",          # 예제 뒤에 추가될 접미사
    input_variables=["input"],      # 입력 변수 지정
)

llm = ChatOpenAI(model="gpt-4o-mini")

# chain = LLMChain(llm=llm, prompt=prompt)
chain = prompt | llm

# query 수정
query_generate = '"2021년 기초학력 진단-보정 시스템 응용S/W 유지관리 및 서비스 운영"의 RFP를 작성해줘'
# print(prompt.invoke({"input": query}).to_string())

response = chain.invoke(query_generate)

# 파일 저장
bid_name = re.sub("\W", "_", re.search(r'"(.+)"', query_generate).group(1))
with open(f"fewshot/output/output_{bid_name}.md", "w", encoding="utf-8") as f:
    f.write(response.content)