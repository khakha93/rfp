from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  

from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid

from langchain_teddynote import logging

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.graphs import visualize_graph

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from nodes import *

load_dotenv()

logging.langsmith("LangGraph-Structures")


# 메모리 저장소 생성
memory = MemorySaver()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini")


# 노드 추가
workflow = StateGraph(GraphState)

workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("generate_basic", lambda x: generate_basic(x, llm))
# workflow.add_node("move_to_next_step", move_to_next_step)
workflow.add_node("check_basic", lambda x: check_basic(x, llm))
workflow.add_node("modify_basic", lambda x: modify_basic(x, llm))
workflow.add_node("generate_requests", lambda x: generate_requests(x, llm))
workflow.add_node("check_requests", lambda x: check_requests(x, llm))
workflow.add_node("modify_requests", lambda x: modify_requests(x, llm))

# 엣지 추가
workflow.add_edge(START, "retrieve_docs")
workflow.add_edge("retrieve_docs", "generate_basic")
workflow.add_edge("generate_basic", "check_basic")
workflow.add_conditional_edges(
    "check_basic",
    move_to_next_step,
    {
        "yes": "generate_requests",
        "no": "modify_basic",
    },
)
workflow.add_edge("modify_basic", "check_basic")
workflow.add_edge("generate_requests", "check_requests")
workflow.add_conditional_edges(
    "check_requests",
    move_to_next_step,
    {
        "yes": END,
        "no": "modify_requests",
    },
)
workflow.add_edge("modify_requests", "check_requests")

# 질문 입력
# inputs = {}
# question = "내 이름은?"
# inputs={"messages": [("user", question)]}

# $$$ 유저 인풋
user_input_title = "기초학력진단평가"
user_input_content = ""
# 질문 입력 $$$ 클래스 변수 변경하기
inputs = GraphState(title=user_input_title, contents=user_input_content)

# 그래프 컴파일
app = workflow.compile(checkpointer=memory)
# 시각화
visualize_graph(app)

# 실행
config = RunnableConfig(recursion_limit=50, configurable={"thread_id": random_uuid()})

invoke_graph(app, inputs, config, ["retrieve_docs", "generate_basic", "prompt_rewrite"])


