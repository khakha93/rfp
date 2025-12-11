def html_to_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    s = s.replace("</div>", "\n").replace("<div>", "")
    s = s.replace("<b>", "").replace("</b>", "")
    s = s.replace("\r\n", "\n")  # ✅ 윈도 개행 보정
    import re as _re
    s = _re.sub(r"<[^>]+>", "", s)
    s = _re.sub(r"\n{3,}", "\n\n", s).strip()
    import html as _html
    s = _html.unescape(s)  # ✅ HTML 엔티티 복원
    return s

def format_req_record(rq_list) -> str:
    # get = lambda k: (rec.get(k) or "").strip()
    # details = (rec.get("details") or "").strip()
    # if not details and rec.get("scope"): details = (rec.get("scope") or "").strip()
    # return "\n".join([
    #     "요구사항 분류: " + get("category"),
    #     "요구사항 고유번호: " + get("code"),
    #     "요구사항 명칭: " + get("title"),
    #     "요구사항 정의: " + get("definition"),
    #     "요구사항 세부내용: " + details
    # ])

    req_text = "\n\n[요구사항 상세]"

    for req in rq_list:
        req_text += (
            f"\n\n- 코드: {req['code']}\n"
            f"- 카테고리: {req['category']}\n"
            f"- 정의: {req['definition']}\n"
            f"- 상세 내용:\n{req['details']}"
        )
    
    return f"--- 요구사항 시작 ---\n{req_text.strip()}\n--- 요구사항 끝 ---"