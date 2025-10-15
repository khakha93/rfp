import json
import os
import re
import pandas as pd
import pytz
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from custom_templates import MM_TEMPLATE



def load_all_json(paths: List[str]) -> List[Document]:
    """
    Load and parse JSON files into a list of Document objects.
    
    Args:
        paths: List of file paths to JSON files
        
    Returns:
        List[Document]: List of Document objects containing the parsed content
    """
    docs: List[Document] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and len(data) == 1:
                entries = next(iter(data.values()))
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            content = (
                                entry.get("definition") or entry.get("summary")
                                or entry.get("details") or entry.get("scope")
                                or entry.get("content") or str(entry)
                            )
                            if content and len(content.strip()) > 10:
                                md = {k: v for k, v in entry.items()
                                      if k not in {"definition", "summary", "details", "scope", "content"}}
                                md["source"] = path
                                docs.append(Document(page_content=str(content).strip(), metadata=md))
        except Exception:
            continue
    return docs


def split_docs(docs: List[Document], chunk_size: int = 400, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        docs: List of Document objects to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of split Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


def detect_topic(user_query: str) -> str:
    """
    Detect the topic of the user query.
    
    Args:
        user_query: The user's query string
        
    Returns:
        str: The detected topic ('mm' for man-month related queries, '' for others)
    """
    q = user_query.lower()
    if "m/m" in q or "맨먼" in q or "man-month" in q or "man month" in q:
        return "mm"
    return ""


def build_mm_block(answer_text: str) -> Tuple[str, str, int]:
    """
    Build a document block for man-month related content.
    
    Args:
        answer_text: The text content for the block
        
    Returns:
        Tuple containing (title, body, stage)
    """
    title = "인력 투입 및 M/M(Man-Month) 산정 기준"
    body = MM_TEMPLATE.format(answer_text=answer_text.strip())
    return title, body, 5  # Default to stage 5


def build_generic_block(user_query: str, answer_text: str) -> Tuple[str, str, int]:
    """
    Build a generic document block.
    
    Args:
        user_query: The original user query
        answer_text: The text content for the block
        
    Returns:
        Tuple containing (title, body, stage)
    """
    title = f"참고 설명: {user_query.strip()[:30]}"
    body = answer_text.strip()
    suggested_stage = resolve_stage_from_text(user_query, 5)  # Default to stage 5
    return title, body, suggested_stage


def resolve_stage_from_text(text: str, default_stage: int = 5) -> int:
    """
    Determine the document stage from the input text.
    
    Args:
        text: The input text to analyze
        default_stage: The default stage to return if no stage is detected
        
    Returns:
        int: The detected stage number (1-5)
    """
    t = text.lower()

    # Keyword priority
    if "사업추진" in t or "추진방안" in t:
        return 4
    if "제안요청" in t or "요청내용" in t:
        return 5

    # Broad pattern matching for stage numbers
    if re.search(r"\b4\s*(?:[.\-:]\s*|단계)\b", t):
        return 4
    if re.search(r"\b5\s*(?:[.\-:]\s*|단계)\b", t):
        return 5

    # Match any single digit number in the text
    m = re.search(r"\b([1-5])\b", t)
    if m:
        return int(m.group(1))

    return default_stage


def detect_apply_command(user_text: str) -> bool:
    """
    Detect if the user text contains a command to apply changes.
    
    Args:
        user_text: The user's input text
        
    Returns:
        bool: True if the text contains an apply command, False otherwise
    """
    t = user_text.lower()
    apply_words = [
        "반영", "추가", "적용", "넣어", "포함",
        "넣어줘", "붙여", "삽입", "기록", "문서화",
        "실어", "반영해", "추가해", "적용해", "포함해"
    ]
    return any(w in t for w in apply_words)


def clean_draft_content(content: str) -> str:
    """
    Clean draft content by removing unwanted patterns and formatting.
    
    Args:
        content: The content to be cleaned
        
    Returns:
        str: The cleaned content with unwanted patterns removed
    """
    if not content: 
        return content
        
    patterns_to_remove = [
        "---\n### ✅ 다음 단계 진행",
        "### ✅ 다음 단계 진행",
        "- **수정이 필요하면**:",
        "- **괜찮다면**:",
        "- **통합본이 필요하면**:",
        "예) '예산을 2억으로 수정'",
        "'다음단계' 또는 '진행' 입력",
        "'최종본' 또는 '통합본' 입력"
    ]
    
    cleaned = content
    for pattern in patterns_to_remove:
        if pattern in cleaned:
            index = cleaned.find(pattern)
            if index != -1:
                cleaned = cleaned[:index].strip()
                break
    
    if cleaned.endswith("---"):
        cleaned = cleaned[:-3].strip()
    
    return cleaned


def save_qa_to_excel(question: str, answer: str, log_path: str) -> None:
    """
    Save Q&A pairs to an Excel file with timestamp.
    
    Args:
        question: The question string
        answer: The answer string
        log_path: Path to the Excel file for saving Q&A logs
        
    Returns:
        None
    """
    try:
        tz = pytz.timezone("Asia/Seoul")
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        row = {"time": now, "question": question, "answer": answer}
        
        if os.path.exists(log_path):
            df_old = pd.read_excel(log_path)
            df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
        else:
            df_new = pd.DataFrame([row])
            
        with pd.ExcelWriter(log_path, engine="openpyxl", mode="w") as writer:
            df_new.to_excel(writer, index=False, sheet_name="log")
    except Exception as e:
        import streamlit as st
        st.warning(f"엑셀 저장 중 오류: {e}")
