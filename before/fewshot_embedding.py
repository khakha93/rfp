import os
import pprint
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI


def wrap_content(content: str) -> str:
    return f"--- 제안요청서 시작 ---\n{content.strip()}\n--- 제안요청서 끝 ---"


# 관련문서 가져오기 (md 파일)
# $$$ 예시로 몇개 가져와서 테스트하기
# $$$ md -> json으로 변경될 예정
related_docs_dir = "C:/kha/project/rfp/data(sample)/recombined_docs"
related_docs = [{"subject": f, "RFP": wrap_content(open(os.path.join(related_docs_dir, f), "r", encoding="utf-8").read())} for f in os.listdir(related_docs_dir) if f.endswith(".md")]


# $$$ Rerank

# 사용자 입력 가져오기
user_input_query = "2021년 기초학력 진단-보정 시스템 응용S/W 유지관리 및 서비스 운영"
user_input_contents = """
"""

# 프롬프트 생성

example_prompt = PromptTemplate(
    input_variables=["subject", "RFP"],
    template="q: '{subject}'의 제안요청서\na: {RFP}\n"
)

# 최종 사용자 입력용 템플릿
suffix_template = """q: '{subject}'의 제안요청서
a:
(단 아래 내용을 포함)
{user_contents}
"""

# FewShotPromptTemplate을 생성합니다.
prompt = FewShotPromptTemplate(
    examples=related_docs,       # 사용할 예제들
    example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
    prefix="다음은 제안요청서 예시입니다.\n",
    suffix=suffix_template,
    input_variables=["subject", "user_contents"],      # 입력 변수 지정
)

# 최종 프롬프트 생성
final_prompt = prompt.format(subject=user_input_query, user_contents=user_input_contents)
print(final_prompt)


# 답변 생성

# LLM 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# chain = LLMChain(llm=llm, prompt=prompt)
chain = prompt | llm

response = chain.invoke(user_input_query)

# 결과 저장
bid_name = re.sub("\W", "_", user_input_query)
with open(f"fewshot/output/output_{bid_name}.md", "w", encoding="utf-8") as f:
    f.write(response.content)