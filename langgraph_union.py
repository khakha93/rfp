from langchain_teddynote import logging
from dotenv import load_dotenv
import nodes

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.graphs import visualize_graph  

from langchain_teddynote.messages import stream_graph, random_uuid
from langchain_core.runnables import RunnableConfig  
from langgraph.checkpoint.memory import MemorySaver


# 환경변수 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-Structures")

# 그래프 정의
workflow = StateGraph(nodes.GraphState)

# 노드 추가
workflow.add_node("query_rewrite", query_rewrite)
# $$$ 리랭크 넣어야 함
workflow.add_node("retrieve_document", retrieve_document)
workflow.add_node("relevance_check", relevance_check)
# $$$ 멀티턴으로
# workflow.add_node("chatbot", chatbot)
# $$$ 이것도 2단계로 나눠야 함
workflow.add_node("llm_answer", llm_answer)

# 노드 연결 (엣지 추가)
workflow.add_edge(START, "query_rewrite")  # 시작 -> 채팅
workflow.add_edge("query_rewrite", "retrieve_document")  # 질문 재작성 -> 검색
workflow.add_edge("retrieve_document", "relevance_check")  # 검색 -> 관련성 체크
# 조건부 엣지
workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.  
    is_relevant,  
    {  
        "yes": "llm_answer",  # 관련성이 있으면 답변을 생성합니다.  
        "no": "query_rewrite",  # 관련성이 없으면 다시 검색합니다.  
    },  
)

workflow.add_edge("llm_answer", END)  # 답변 -> 종료


# 그래프 진입점 설정
workflow.set_entry_point("query_rewrite")
# 체크포인터 설정
memory = MemorySaver()
# 그래프 컴파일
app = workflow.compile(checkpointer=memory)
# 그래프 시각화
visualize_graph(app)


# 그래프 실행
# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})


# $$$ 유저 인풋
user_input_title = ""
user_input_content = ""

# 질문 입력 $$$ 클래스 변수 변경하기
inputs = nodes.GraphState(title=user_input_title, contents=user_input_content)

# 그래프 실행
stream_graph(app, inputs, config, ["query_rewrite", "llm_answer"])

outputs = app.get_state(config).values
print(outputs)
