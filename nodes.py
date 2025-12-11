from typing import TypedDict, Annotated
from httpx import request
from langgraph.graph import add_messages
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from templates import SYSTEM_PROMPT, SYSTEM_PROMPT_2, SUFFIX_TEMPLATE, SUFFIX_TEMPLATE_2
import retrieve_module as rm
from text_processing import format_req_record

import os
import tiktoken


MAX_HISTORY = 10
MAX_EXAMPLES = 2


# GraphState 상태 정의
class GraphState(TypedDict):
    title: Annotated[str, "Title"]  # 사업명  
    contents: Annotated[str, "Contents"]  # 사업내용
    include_contents: Annotated[str, "Include Contents"]  # 사업내용
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    bid_ids: Annotated[list, "Bid IDs"]  # 비딩 넘버 리스트
    rfp_basic: Annotated[str, "RFP Basic"]  # 제안요청서 기본정보
    rfp_requests: Annotated[str, "RFP Requests"]  # 제안요청서 요구사항
    answer: Annotated[str, "Answer"]  # 답변  
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)  
    relevance: Annotated[str, "Relevance"]  # 관련성
    related_docs: Annotated[str, "Related Docs"]  # 관련 문서


def retrieve_docs(state):
    print("retrieve_docs 실행")
    def wrap_content(content: str) -> str:
        return f"--- 제안요청서 시작 ---\n\n{content.strip()}\n\n--- 제안요청서 끝 ---"

    retrive_str = f"{state['title']}\n\n{state['include_contents']}"
    a_results = rm.retrieve_docs(retrive_str, k=10, mode="A")

    # bid_num_list = list(dict.fromkeys([d.metadata['raw']['id'] for d in a_results]))

    result_ids, result_names, result_docs = rm.format_A_results(a_results)

    print()
    for id, name in zip(result_ids, result_names):
        print(f"ID: {id} | Name: {name}")
    print()
    selected_ids = input("\n문서 선택: ").split()
    user_input_content = input("\n추가 사항: ")
    state['contents'] = user_input_content

    formatted_contexts = [{"RFP": wrap_content(doc), 'subject': name} for name, doc, id in zip(result_names, result_docs, result_ids) if id in selected_ids]

    # print(_html_to_text(a_text))
    return GraphState(context=formatted_contexts, bid_ids=selected_ids, contents=user_input_content)

def generate_basic(state, llm):
    print("generate_basic 실행")

    example_prompt = PromptTemplate(
        input_variables=["subject", "RFP"],
        template="q: '{subject}'의 제안요청서\na: {RFP}\n"
    )

    # $$$
    # a_text = "\n<br><br>\n".join(result_docs)

    # FewShotPromptTemplate을 생성합니다.
    prompt = FewShotPromptTemplate(
        examples=state['context'],       # 사용할 예제들
        example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
        # prefix="다음은 제안요청서 예시입니다.\n",
        prefix=SYSTEM_PROMPT,
        suffix=SUFFIX_TEMPLATE,
        input_variables=["subject", "user_contents"],      # 입력 변수 지정
    )

    llm_chain = prompt | llm

    # 체인을 호출하여 답변을 생성합니다. $$$ history 부분 추가
    response = llm_chain.invoke(  
        {  
            "subject": state['title'],
            "user_contents": state['contents'],
            "chat_history": state["messages"],
        }
    )
    # return GraphState(messages=response)
    state['messages'] = response
    return state

def check_basic(state, llm):
    print("check_basic 실행")
    # 제안요청서 기본정보 부분 최종확인 $$$ 히스토리 기반으로 수정하기
    question = "지금까지 수정한 내용을 적용한 최종 제안요청서 전체를 출력해줘. (수정한 내용이 없다면 최초 생성한 내용을 출력)"
    state['messages'].append(("user", question))
    response = llm.invoke(state['messages'])
    state['messages'].append(("assistant", response.content))

    state['rfp_basic'] = response.content
    print(state['rfp_basic'])
    # print(state['messages'][-1].content)
    return state

def modify_basic(state, llm):
    print("modify_basic 실행")

    question = input("질문 다시 입력: ")
    state['messages'].append(("user", question))
    response = llm.invoke(state['messages'])
    state['messages'].append(("assistant", response.content))

    # return GraphState(messages=state['messages'])
    return state

def move_to_next_step(state):
    print("move_to_next_step 실행")
    res = ["yes", "no"][int(input("다음 단계로 넘어갈까요? Yes: 1, No: 2\n"))-1]
    # print(f"결과: {res}")
    return res

def generate_requests(state, llm):
    print("generate_requests 실행")
    # $$$ 기본 정보 -> 요구사항 퓨샷
    # 제안요청서 기본 정보
    rfp_basic = state['rfp_basic']
    # 요구사항 퓨샷: 뭐랑 뭐를 퓨샷하지?
    # 가장 잘 된 문서 1개 정도 -> basic(subject 포함) / requests 분리
    # 
    example_prompt = PromptTemplate(
        input_variables=["ex_num", "basic", "requests"],
        template="""<<예시 {ex_num}번>>

[[프로젝트 정보]]
{basic}


[[요구사항 총괄표]]
{requests}

----------------------------------------------------------------------
""")
    
    examples = []
    idx = 1
    for bid_id, basic in zip(state['bid_ids'], state['context']):
        request_list = rm.retrieve_docs(bid_id, mode="REQ", bid=bid_id)[0].metadata['items']
        examples.append({"ex_num": idx, "basic": basic['RFP'].strip(), "requests": format_req_record(request_list)})
        idx += 1

    examples = examples[:MAX_EXAMPLES]

    # FewShotPromptTemplate을 생성합니다.
    prompt = FewShotPromptTemplate(
        examples=examples,       # 사용할 예제들
        example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
        prefix=SYSTEM_PROMPT_2,
        suffix=SUFFIX_TEMPLATE_2,
        input_variables=["basic", "user_contents"],      # 입력 변수 지정
    )

    llm_chain = prompt | llm

    # 체인을 호출하여 답변을 생성합니다.
    response = llm_chain.invoke(  
        {  
            "basic": state['rfp_basic'],
            "user_contents": state['contents'],
            # "chat_history": state["messages"],
        }
    )
    
    # $$$ 프롬프트 확인용
    sample_input = {
    "basic": state["rfp_basic"],
    "user_contents": state["contents"]
    }

    prompt_text = prompt.format(**sample_input)
    file_path = "/media/HDD/RFP_project/sh/develop_sh/prompt.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    # return GraphState(messages=response)
    state['messages'] = response
    return state

def check_requests(state, llm):
    print("check_requests 실행")

    # 잘 넘어왔는지 확인

    question = "생성 및 수정한 요구사항 총괄표 전체를 출력해줘. 맨 앞에는 요약한 표를 추가해줘."
    state['messages'].append(("user", question))
    response = llm.invoke(state['messages'])
    state['messages'].append(("assistant", response.content))

    state['rfp_requests'] = response.content
    print(state['rfp_basic'], state['rfp_requests'], sep="\n\n")
    # 제안요청서 요구사항 부분 최종 확인 $$$
    # print(state['messages'][-1].content)
    return state

def modify_requests(state, llm):
    print("modify_requests 실행")

    question = input("질문 다시 입력: ")
    state['messages'].append(("user", question))
    response = llm.invoke(state['messages'])
    state['messages'].append(("assistant", response.content))

    return state