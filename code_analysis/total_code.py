import os
import re
import json
import uuid
from typing import List, Tuple
from datetime import datetime
import custom_templates as templates
from stage_manager import ImprovedRFPStageManager

import streamlit as st
import pandas as pd
# import pytz
# import torch
from dotenv import load_dotenv

# =========================
# 환경 변수 및 기본 설정
# =========================
from config import DATA_FILES, PERSIST_DIR, COLLECTION, LOG_PATH, DEVICE

load_dotenv(dotenv_path='C:\\kha\\project\\rfp\\code_analysis\\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")

# =========================
# 설정 임포트
# =========================

st.set_page_config(page_title="제안요청서 AI 초안 작성 챗봇", page_icon="📋", layout="centered")
st.markdown("### 제안요청서 AI 초안 작성 챗봇")

with st.expander("💡 사용 가이드", expanded=False):
    st.markdown(templates.USAGE_GUIDE)

try:
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI
    # OpenAI Embeddings (기존 유지)
    from .openai_embeddings import OpenAIEmbeddings
    # 문서 로딩/분할 (기존 유지)
    from .document_utils import load_all_json, split_docs
    # Q/A 엑셀 저장 유틸 (누락 복구)
    from .document_utils import save_qa_to_excel
    # 기존 유틸/분류 로직 (원형 유지)
    from .text_processing import (
        extract_user_mentioned_functions,
        extract_requirements_from_input,
        is_final_command,
        is_requirement_related,
        is_progression_command,
        classify_user_input
    )
    # (추가) 일반 질문 → 문서 반영 지원
    from .document_utils import (
        detect_topic,
        build_mm_block,
        build_generic_block,
        resolve_stage_from_text,
        detect_apply_command,
        clean_draft_content
    )
except ImportError as e:
    st.error(f"필수 라이브러리 누락: {e}")
    st.stop()

# =========================
# VectorStore (기존 유지)
# =========================
@st.cache_resource(show_spinner=True)
def bootstrap_vectorstore() -> Tuple["Chroma", int, bool]:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 필요합니다!")
        st.stop()
    embedding = OpenAIEmbeddings()
    os.makedirs(PERSIST_DIR, exist_ok=True)
    try:
        vs = Chroma(collection_name=COLLECTION, embedding_function=embedding,
                    persist_directory=PERSIST_DIR)
        count = vs._collection.count()
        if count and count > 0:
            return vs, count, False
    except Exception:
        pass
    docs = load_all_json(DATA_FILES)
    if not docs:
        st.error("로드된 문서가 없습니다! 데이터 파일 경로를 확인해주세요.")
        st.stop()
    chunks = split_docs(docs)
    vs = Chroma(collection_name=COLLECTION, embedding_function=embedding,
                persist_directory=PERSIST_DIR)
    for i in range(0, len(chunks), 50):
        vs.add_documents(chunks[i:i+50])
    vs.persist()
    count = vs._collection.count()
    return vs, count, True

# =========================
# 단계/상태 (기존 유지 + 최소 필드 추가)
# =========================

def _new_chat(title="새 대화"):
    cid = uuid.uuid4().hex[:8]
    st.session_state.convos[cid] = {"title": title, "messages": [], "history": []}
    st.session_state.current_chat_id = cid
    st.session_state["__pending_select"] = cid
    st.session_state.stage = 1
    st.session_state.collected_info = {}
    st.session_state.stage_drafts = {}
    # === 추가: 일반 질문 반영 ===
    st.session_state.stage_inserts = {}   # {stage: [{"title","body"}]}
    st.session_state.pending_block = None # {"title","body","stage"}
    return cid

def _ensure_state():
    if "convos" not in st.session_state: st.session_state.convos = {}
    if "current_chat_id" not in st.session_state: _new_chat("새 대화")
    if "stage" not in st.session_state: st.session_state.stage = 1
    if "collected_info" not in st.session_state: st.session_state.collected_info = {}
    if "stage_drafts" not in st.session_state: st.session_state.stage_drafts = {}
    # === 추가: 일반 질문 반영 ===
    if "stage_inserts" not in st.session_state: st.session_state.stage_inserts = {}
    if "pending_block" not in st.session_state: st.session_state.pending_block = None
    # 세션 상태 기본값 (없으면 만들어 둠)
    if "pending_block" not in st.session_state:
        st.session_state.pending_block = None
    if "stage_inserts" not in st.session_state:
        st.session_state.stage_inserts = {}
    if "last_explanation" not in st.session_state:
        st.session_state.last_explanation = None


def _current():
    _ensure_state()
    cid = st.session_state.current_chat_id
    return cid, st.session_state.convos[cid]

def _delete_chat(cid: str):
    if "convos" not in st.session_state: return
    if cid in st.session_state.convos: del st.session_state.convos[cid]
    if st.session_state.convos:
        new_cid = next(iter(st.session_state.convos.keys()))
        st.session_state.current_chat_id = new_cid
        st.session_state["__pending_select"] = new_cid
    else:
        new_cid = _new_chat("새 대화"); st.session_state["__pending_select"] = new_cid


# =========================
# (추가) 일반 질문 → 문서 반영 지원
# =========================

def insert_block_into_stage(stage: int, block_title: str, block_body: str):
    inserts = st.session_state.stage_inserts.get(stage, [])
    inserts.append({"title": block_title, "body": block_body})
    st.session_state.stage_inserts[stage] = inserts


# =========================
# RAG/생성 (기존 유지)
# =========================
def build_rag_stage_prompt(stage: int, collected_info: dict, user_query: str) -> str:
    sm = ImprovedRFPStageManager()
    si = sm.get_stage_info(stage)
    confirmed = ""
    if collected_info:
        confirmed = "\n### 📋 기존 수집 정보\n" + "\n".join(
            f"- **{k}**: {v}" for k, v in collected_info.items() if v and v != "[미정]"
        )
    base = template.RAG_BASE.format(stage=stage, title=si['title'], description=si['description'], confirmed=confirmed)
    if stage == 1:
        base += template.RAG_1ST
    elif stage == 2:
        base += template.RAG_2ND
    elif stage == 3:
        base += template.RAG_3RD
    elif stage == 4:
        base += template.RAG_4TH
    elif stage == 5:
        base += template.RAG_5TH
    return base

def generate_rag_based_answer(query: str, stage: int, collected_info: dict, vs: "Chroma") -> str:
    if not OPENAI_API_KEY:
        return template.NO_API_KEY.format(stage=stage)
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
        prompt = build_rag_stage_prompt(stage, collected_info, query)

        enhanced_prompt = template.RAG_PROMPT.format(prompt=prompt, query=query)
        return llm.invoke(enhanced_prompt).content
    except Exception as e:
        return ERROR_1ST.format(stage=stage, e=str(e))

# =========================
# 일반 질문 처리 (기존 함수에 최소 변경)
# =========================
def handle_general_question(user_query: str, vs: "Chroma") -> str:
    # ✅ 항상 먼저 토픽 계산
    topic = detect_topic(user_query)

    # 답변 생성
    if not OPENAI_API_KEY:
        answer = (
            "M/M(Man-Month)은 1인이 1개월간 풀타임으로 수행하는 작업량 단위입니다."
            if topic == "mm" else
            "해당 질문에 대한 간단한 답변입니다."
        )
    else:
        try:
            docs = vs.similarity_search(user_query, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
            prompt = template.RAG_QA_PROMPT.format(context=context, user_query=user_query)
            answer = llm.invoke(prompt).content
        except Exception as e:
            answer = f"오류가 발생했습니다: {str(e)}"
    
    # 안내 문구(일반화)
    answer += "\n\n— 이 **설명**을 문서에 반영하려면 '반영/추가/적용/넣어/포함' 중 하나만 입력하세요. (특정 단계면 '4단계/5단계' 함께 적기)"
    return answer


# =========================
# 초안 정리/통합 (기존 유지 + 삽입 블록 병합)
# =========================

def build_full_compilation(vs: "Chroma") -> str:
    # 기존 초안 확보 (기존 로직 유지)
    for s in range(1,6):
        if s not in st.session_state.stage_drafts:
            fallback_query = "표준으로 작성" if s in (2,3,4,5) else "표준"
            draft = generate_rag_based_answer(fallback_query, s, st.session_state.collected_info, vs)
            st.session_state.stage_drafts[s] = draft
    compiled = []
    titles = {1:"## 1. 기본 정보",2:"## 2. 사업개요",3:"## 3. 시스템 현황",4:"## 4. 사업추진방안",5:"## 5. 제안요청내용"}
    for s in range(1,6):
        content = clean_draft_content(st.session_state.stage_drafts[s])
        section = titles[s] + "\n\n" + content
        # === 추가: 일반 질문 반영 블록 ===
        inserts = st.session_state.stage_inserts.get(s, [])
        if inserts:
            section += "\n\n### 추가 반영 사항\n"
            for blk in inserts:
                section += f"\n#### {blk['title']}\n{blk['body']}\n"
        compiled.append(section)
    final_compilation = template.FULL_COMPILED.format(compiled=chr(10).join(f"{content}" for content in compiled))
    return final_compilation

# =========================
# 입력 처리 (기존 유지 + 반영 커맨드만 추가)
# =========================
def process_user_input(user_text: str, vs: "Chroma"):
    # 0) '반영/추가/적용/...' 우선 처리
    if detect_apply_command(user_text):
        # A) pending_block이 있으면 그걸 반영
        if st.session_state.pending_block:
            blk = st.session_state.pending_block
            stage = resolve_stage_from_text(user_text, blk.get("stage", 5))
            insert_block_into_stage(stage, blk["title"], blk["body"])
            st.session_state.pending_block = None
            return f"✅ **'{blk['title']}'** 내용을 **{stage}단계**에 반영했습니다. '최종본'에서 확인하세요."

        # B) pending이 없어도, 방금 직전 설명이 있으면 그걸 반영
        if st.session_state.last_explanation:
            blk = st.session_state.last_explanation.copy()
            stage = resolve_stage_from_text(user_text, blk.get("stage", 5))
            insert_block_into_stage(stage, blk["title"], blk["body"])
            st.session_state.pending_block = None
            return f"✅ **'{blk['title']}'**(을)를 **{stage}단계**에 반영했습니다. '최종본'에서 확인하세요."

        # C) 그래도 없으면 토픽 감지로 최소 생성 후 반영(M/M 등)
        topic = detect_topic(user_text)
        if topic == "mm":
            base = "M/M(Man-Month)은 1인이 1개월간 풀타임으로 수행하는 작업량 단위입니다."
            title, body, s = build_mm_block(base)
            stage = resolve_stage_from_text(user_text, s)
            insert_block_into_stage(stage, title, body)
            return f"✅ **'{title}'** 내용을 **{stage}단계**에 반영했습니다. '최종본'에서 확인하세요."

        # D) 정말 없으면 톤을 부드럽게
        return "지금은 바로 반영할 설명이 없네요. 먼저 간단히 물어보세요. 예: “M/M 설명해줘” → “반영”"

    # (이하 기존 로직과 동일)
    input_type = classify_user_input(user_text)

    if input_type == 'general_question':
        answer = handle_general_question(user_text, vs)
        return answer

    if input_type == 'modification':
        # 기존 수정 처리 (원본 유지)
        return "어떤 항목을 어떻게 수정할지 구체적으로 말씀해 주세요. 예: '예산을 3억원으로 수정'"

    if input_type == 'final':
        return build_full_compilation(vs)

    if input_type == 'progression':
        st.session_state.stage = min(st.session_state.stage + 1, 5)
        si = ImprovedRFPStageManager().get_stage_info(st.session_state.stage)
        return f"**{st.session_state.stage}/5단계 - {si['title']} 진행합니다.**\n\n{si['initial_question']}"

    # 현재 단계 초안 생성 (원본 유지)
    stage = st.session_state.stage
    draft = generate_rag_based_answer(user_text, stage, st.session_state.collected_info, vs)
    st.session_state.stage_drafts[stage] = draft
    return draft

# =========================
# 앱 렌더링 (기존 유지)
# =========================
def main():
    # 임베딩 초기화
    try:
        vs, doc_count, was_built = bootstrap_vectorstore()
    except Exception as e:
        st.error(f"⚠ 벡터스토어 초기화 실패: {e}")
        st.error("데이터 파일 경로를 확인하거나 OpenAI API 키를 설정해주세요.")
        st.stop()

    # UI 렌더링
    with st.sidebar:
        st.markdown("---")
        if st.button("➕ 새 채팅", use_container_width=True):
            _new_chat("새 대화"); st.rerun()
        st.divider()

        # 단계 정보 표시
        sm=ImprovedRFPStageManager()
        si=sm.get_stage_info(st.session_state.stage)
        st.markdown(f"**현재 단계: {st.session_state.stage}/5**")
        st.markdown(f"*{si['title']}*")
        st.caption(si['description'])
        st.progress((st.session_state.stage-1)/4)

        # $$$ 여기서부터 다시 보기
        chat_ids=list(st.session_state.convos.keys())
        if not chat_ids:
            _new_chat("새 대화"); chat_ids=list(st.session_state.convos.keys())
        pending=st.session_state.pop("__pending_select", None)
        if pending:
            st.session_state.current_chat_id=pending; st.session_state["chat_selector"]=pending
        if len(chat_ids)>1:
            selected=st.radio("대화 선택", options=chat_ids,
                              format_func=lambda x: st.session_state.convos[x]["title"],
                              label_visibility="collapsed", key="chat_selector")
            if selected: st.session_state.current_chat_id=selected
            st.divider()
            confirm=st.checkbox("삭제 확인", key="confirm_del")
            if st.button("선택된 채팅 삭제", use_container_width=True, disabled=not confirm):
                _delete_chat(selected); st.rerun()

    cur_id, cur = _current()
    cur.setdefault("messages", []); cur.setdefault("history", [])

    seed_key=f"_seeded_{cur_id}"
    if not st.session_state.get(seed_key, False) and len(cur["messages"])==0:
        si=ImprovedRFPStageManager().get_stage_info(st.session_state.stage)
        msg = template.WELCOME_MESSAGE.format(stage=st.session_state.stage, title=si['title'], initial_question=si['initial_question'])
        cur["messages"].append({"role":"assistant","content":msg})
        st.session_state[seed_key]=True

    for m in cur["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("답변을 입력해주세요...", key="main_input")

    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("작성 중..."):
                answer = process_user_input(user_text, vs)
                st.markdown(answer)
                try:
                    save_qa_to_excel(user_text, answer, LOG_PATH)
                except Exception as e:
                    st.warning(f"엑셀 저장 중 오류: {e}")

        cur["messages"].append({"role":"user","content":user_text})
        cur["messages"].append({"role":"assistant","content":answer})

if __name__ == "__main__":
    _ensure_state()
    main()
