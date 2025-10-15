from langchain_openai import ChatOpenAI  
from langgraph.graph import END, StateGraph
from langchain_teddynote.evaluator import GroundednessChecker  
from langchain_teddynote.messages import messages_to_history
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini")


# GraphState 상태 정의
class GraphState(TypedDict):
    title: Annotated[str, "Title"]  # 사업명  
    contents: Annotated[str, "Contents"]  # 사업내용  
    question: Annotated[str, "Question"]  # 질문  
    context: Annotated[str, "Context"]  # 문서의 검색 결과  
    answer: Annotated[str, "Answer"]  # 답변  
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)  
    relevance: Annotated[str, "Relevance"]  # 관련성
    related_docs: Annotated[str, "Related Docs"]  # 관련 문서


def chatbot(state: GraphState) -> GraphState:
    return {"messages": [llm.invoke(state["messages"])]}

# $$$ 관련 문서 가져오기 (임시)
def get_related_docs(question):
    related_docs_dir = "C:/kha/project/rfp/data(sample)/recombined_docs"
    related_docs = [{"subject": f, "RFP": wrap_content(open(os.path.join(related_docs_dir, f), "r", encoding="utf-8").read())} for f in os.listdir(related_docs_dir) if f.endswith(".md")]
    return related_docs
def format_docs(docs):
    return docs


# 문서 검색 노드  
def retrieve_document(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옵니다.  
    latest_question = state["question"]  

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.  
    retrieved_docs = get_related_docs(latest_question)  
    # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)  
    retrieved_docs = format_docs(retrieved_docs)  

    # 검색된 문서를 context 키에 저장합니다.  
    return GraphState(context=retrieved_docs)

# 답변 생성 노드  
def llm_answer(state: GraphState) -> GraphState:
    def wrap_content(content: str) -> str:
        return f"--- 제안요청서 시작 ---\n{content.strip()}\n--- 제안요청서 끝 ---"
    
    # $$ 파일 형식 나오면 조정 필요
    related_docs = [{"subject": doc["subject"], "RFP": wrap_content(doc["RFP"])} for doc in state["related_docs"]]

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

    llm_chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 체인을 호출하여 답변을 생성합니다. $$$ history 부분 추가
    response = llm_chain.invoke(  
        {  
            "subject": state['title'],  
            "user_contents": state['contents'],
            "chat_history": messages_to_history(state["messages"]),
        }
    )

    # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.  
    return GraphState(  
        answer=response, messages=[("user", latest_question), ("assistant", response)]  
    )

# 관련성 체크 노드  
def relevance_check(state: GraphState) -> GraphState:  
    # 관련성 평가기를 생성합니다.  
    question_answer_relevant = GroundednessChecker(  
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), target="question-retrieval"  
    ).create()

    # 관련성 체크를 실행("yes" or "no")  
    response = question_answer_relevant.invoke(  
        {"question": state["question"], "context": state["context"]}  
    )  

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)  

    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!  
    return GraphState(relevance=response.score)

# 관련성 체크하는 함수(router)  
def is_relevant(state: GraphState) -> GraphState:  
    return state["relevance"]

# Query Rewrite 노드
def query_rewrite(state: GraphState) -> GraphState:
    latest_question = state["question"][-1].content
    question_rewriter = (
        re_write_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    )
    question_rewritten = question_rewriter.invoke({"question": latest_question})
    return GraphState(question=question_rewritten)