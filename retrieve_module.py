from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document


def format_A_results(docs: List[Document]) -> str:
    result_docs = []
    result_ids = []
    result_names = []

    for d in docs:
        meta = d.metadata or {}
        raw = meta.get("raw", meta)

        bid = str(raw.get("id", "")).strip()

        name = raw.get("bidding_name", "")


        result_docs.append(
            "사업명: " + " " +
            "비딩넘버: " + " " +
            "사업범위: " + " " +
            "대분류: " + " " +
            "중분류: " + " " +
            "세부분류: " + " " +
            "분류번호: " + " " +
            "배경: " + " " +
            "영향: " + " " +
            "요약: " + " "
        )

        result_ids.append(bid)
        result_names.append(name)

    return result_ids, result_names, result_docs


def format_req_record(rq_list) -> str:

    req_text = "\n\n[요구사항 상세]"

    for req in rq_list:
        req_text += (
            f"\n\n- 코드: {req['code']}\n"
            f"- 카테고리: {req['category']}\n"
            f"- 정의: {req['definition']}\n"
            f"- 상세 내용:\n{req['details']}"
        )
    
    return f"--- 요구사항 시작 ---\n{req_text.strip()}\n--- 요구사항 끝 ---"