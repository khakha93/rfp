"""
Text processing utilities for RFP generation.

This module contains various utility functions for text analysis, classification,
and processing that are used throughout the RFP generation process.
"""

from typing import List, Dict, Any, Set


def extract_user_mentioned_functions(user_input: str) -> List[str]:
    """
    Extract function-related terms from user input based on common Korean suffixes.
    
    Args:
        user_input: The input text to analyze
        
    Returns:
        List of extracted function terms (up to 5)
    """
    inds = ["기능", "처리", "관리", "제공", "지원", "연동", "분석", "생성"]
    words = user_input.replace(",", " ").replace(".", " ").split()
    out = []
    
    for i, w in enumerate(words):
        # Check for suffixes in individual words
        for ind in inds:
            if w.endswith(ind) and len(w) > len(ind):
                base = w[:-len(ind)]
                if len(base) >= 2:
                    out.append(f"{base} {ind}")
        
        # Check for two-word combinations
        if i < len(words) - 1:
            comb = f"{w} {words[i+1]}"
            if any(ind in comb for ind in inds):
                out.append(comb)
    
    return out[:5]


def extract_requirements_from_input(user_query: str, collected_info: dict) -> List[str]:
    """
    Extract relevant requirements based on user query and collected information.
    
    Args:
        user_query: The current user query
        collected_info: Dictionary of previously collected information
        
    Returns:
        List of extracted requirements (up to 8 unique items)
    """
    full = f"{collected_info.get('사업명','')} {collected_info.get('사업목적','')} {user_query}"
    
    # Define requirement patterns
    patterns = {
        "로그인": ["사용자 인증", "로그인 기능", "접근 제어"],
        "회원": ["회원 가입", "회원 관리", "프로필 관리"],
        "검색": ["통합 검색", "고급 검색", "검색 결과 정렬"],
        "게시": ["게시글 작성", "게시글 관리", "댓글 기능"],
        "파일": ["파일 업로드", "파일 다운로드", "첨부파일 관리"],
        "알림": ["푸시 알림", "이메일 알림", "SMS 알림"],
        "리포트": ["보고서 생성", "통계 분석", "데이터 시각화"],
        "백업": ["데이터 백업", "복구 기능", "이중화"],
        "API": ["외부 API 연동", "REST API 제공", "웹훅 지원"],
        "챗봇": ["자연어 처리", "대화 관리", "FAQ 자동 응답", "학습 기능"],
        "AI": ["인공지능 분석", "머신러닝", "자동 분류", "예측 기능"],
        "모바일": ["모바일 앱", "반응형 웹", "터치 인터페이스"],
        "관리": ["시스템 관리", "사용자 관리", "권한 관리", "설정 관리"],
        "모니터링": ["시스템 모니터링", "로그 분석", "성능 추적"],
        "연동": ["시스템 연동", "데이터 동기화", "인터페이스"],
        "워크플로우": ["업무 프로세스", "승인 절차", "단계별 처리"],
        "대시보드": ["실시간 대시보드", "KPI 표시", "차트 및 그래프"],
    }
    
    # Find matching patterns
    out = []
    low = full.lower()
    for k, funcs in patterns.items():
        if k in low:
            out.extend(funcs)
    
    # Add any user-mentioned functions
    out.extend(extract_user_mentioned_functions(user_query))
    
    # Remove duplicates while preserving order
    unique = list(dict.fromkeys(out))
    return unique[:8] if unique else ["기본 기능", "사용자 인터페이스", "데이터 처리"]


def is_final_command(text: str) -> bool:
    """Check if the text indicates a request for the final document."""
    toks = {"최종본", "통합본", "전체초안", "전체 통합본", "1~5단계", "전체"}
    t = text.lower().strip()
    return any(k in t for k in toks)


def is_requirement_related(user_query: str) -> bool:
    """Check if the query is related to requirements gathering."""
    keys = [
        "요구사항", "기능요구사항", "기능", "성능", "보안", "품질", 
        "제약", "인터페이스", "데이터", "테스트", "프로젝트지원", 
        "요구조건", "필요사항", "스펙", "specification", "requirement"
    ]
    ul = user_query.lower()
    return any(k in ul for k in keys)


def is_progression_command(text: str) -> bool:
    """Check if the text indicates a command to proceed to the next step."""
    toks = {
        "다음", "다음단계", "진행", "넘어가", "계속", 
        "ok", "좋아", "맞아", "다음으로", "넥스트", "next", "완료"
    }
    t = text.lower().replace(" ", "")
    return any(k in t for k in toks)


def classify_user_input(user_text: str) -> str:
    """
    Classify the user's input into different categories.
    
    Args:
        user_text: The text input from the user
        
    Returns:
        str: Classification result ('progression', 'final', 'modification', 'question', or 'general')
    """
    text_lower = user_text.lower().strip()
    
    if is_progression_command(user_text):
        return 'progression'
    if is_final_command(user_text):
        return 'final'
    
    # Check for modification keywords
    modification_keywords = [
        '수정', '변경', '바꿔', '고쳐', '조정', '늘려', '줄여', '올려', '내려',
        '추가', '제거', '삭제', '포함', '업데이트'
    ]
    if any(keyword in text_lower for keyword in modification_keywords):
        return 'modification'
    
    # Check for question indicators
    question_indicators = [
        '?', '뭐', '무엇', '어떻게', '왜', '언제', '어디서', '누가', '어떤',
        '설명', '알려줘', '궁금', '문의', '질문', '도움', '방법', '차이', 'm/m', '맨먼', 'man-month'
    ]
    if any(indicator in text_lower for indicator in question_indicators):
        return 'question'
    
    return 'general'
