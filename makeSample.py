# RFP 샘플 데이터 생성 및 전처리 스크립트
#
# 이 스크립트는 다음과 같은 기능을 수행합니다:
# 1. TSV 파일에서 RFP 관련 데이터를 불러오기
# 2. 데이터 전처리 및 정제
# 3. JSON 형식으로 데이터 통합
# 4. 마크다운 형식으로 최종 RFP 문서 생성
#
# 주요 의존성:
# - pandas
# - tqdm

import pandas as pd
import os
from tqdm import tqdm


dir_path = r"C:\kha\project\rfp\data(sample)"
inter_dir = r"C:\kha\project\rfp\fewshot\inter_results"
save_dir = r"C:\kha\project\rfp\fewshot\answers"
f_names = [os.path.join(dir_path,f) for f in os.listdir(dir_path)]
df_list = [pd.read_csv(f, sep="\t") for f in f_names]


def get_head_infos(bid_id:int):
    bidding_id, title = df_list[0][df_list[0]['bid_no'] == bid_id].iloc[0][['bidding_id', 'name']]
    bidding_info_id = df_list[2][df_list[2]['bidding_id'] == bidding_id].iloc[0, 0]
    return bidding_info_id, title

def get_requests(bidding_info_id:int) -> pd.DataFrame:
    rq = df_list[3][df_list[3]['bidding_info_id'] == bidding_info_id]
    rq_ref = rq[['code', 'category', 'definition', 'details']].sort_values(by=['category', 'code'])

    # 카테고리 결측치 채우기
    category_dict = {}
    for row in rq_ref[['code', 'category']].dropna().itertuples():
        code_front = row.code.split('-')[0]
        if code_front not in category_dict:
            category_dict[code_front] = row.category
    
    # rq_ref['category'] = rq_ref['code'].apply(lambda x: category_dict[])
    rq_ref['category'] = rq_ref['code'].apply(lambda x: category_dict.get(x.split('-')[0], '-'))
    rq_ref = rq_ref.fillna('-').sort_values(by=['category', 'code'])

    return rq_ref

def merge_infos(bid_id:int):
    bidding_info_id, title = get_head_infos(bid_id)

    target_doc = df_list[1][df_list[1]['bidding_info_id'] == bidding_info_id]
    summary, scope = target_doc[['summary', 'scope']].iloc[0]
    df_basic_info = pd.DataFrame(columns=['title', 'summary', 'scope'], data=[[title, summary, scope]])

    df_rq_ref = get_requests(bidding_info_id)

    df_basic_info.to_csv(os.path.join(inter_dir, f'{bid_id}_basic_info.csv'))
    df_rq_ref.to_csv(os.path.join(inter_dir, f'{bid_id}_requests.csv'))

    return title, df_rq_ref

def merge_2_json(bid_id):
    # 1. basic_info.csv 읽기
    df_info = pd.read_csv(os.path.join(inter_dir, f'{bid_id}_basic_info.csv'))
    df_info['scope'] = df_info['scope'].fillna('')
    # 2. requests.csv 읽기
    df_requests = pd.read_csv(os.path.join(inter_dir, f'{bid_id}_requests.csv'))

    # 3. basic_info 가공
    basic_info = {"project_name": df_info.loc[0, "title"],
                  "summary": df_info.loc[0, "summary"],
                  "scope": [line.strip(" *") for line in df_info.loc[0, "scope"].split("\n") if line.strip()]}

    # 4. 요구사항 리스트 가공
    requirements = []
    for _, row in df_requests.iterrows():
        item = {"code": row["code"],
                "category": row["category"],
                "definition": row["definition"],
                "details": row["details"].strip()}
        requirements.append(item)

    # 5. JSON 형태로 병합
    merged_data = {"basic_info": basic_info,
                   "requirements": requirements}

    # 6. JSON 파일 저장
    file_path = os.path.join(inter_dir, f"{bid_id}_merged.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    # print(f"✅ JSON 병합 완료: {file_path}")
    return file_path

def refine_2_md(bid_id):
    # JSON 파일 로드 및 문자열 변환
    with open(os.path.join(inter_dir, f"{bid_id}_merged.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        # ans_01 = json.dumps(ans_01, ensure_ascii=False, indent=2)

    # 문자열 포맷으로 변환
    basic_info = data["basic_info"]
    requirements = data["requirements"]

    # 기본정보 포맷
    basic_text = (
        f"[프로젝트명]\n{basic_info['project_name']}\n\n"
        f"[개요]\n{basic_info['summary']}\n\n"
        f"[범위]\n" + "\n".join(f"- {item}" for item in basic_info["scope"])
    )

    # 요구사항 포맷
    req_text = "\n\n[요구사항 상세]"
    for req in requirements:
        req_text += (
            f"\n\n- 코드: {req['code']}\n"
            f"- 카테고리: {req['category']}\n"
            f"- 정의: {req['definition']}\n"
            f"- 상세 내용:\n{req['details']}"
        )

    # 전체 답변 텍스트
    full_answer = f"{basic_text}\n\n{req_text}"

    # 파일 저장
    file_path = os.path.join(save_dir, f"{bid_id}_refined.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_answer)
    # print(f"✅ md로 정제 완료: {file_path}")
    return file_path


if __name__=='__main__':
    bid_ids = df_list[0].bid_no.to_list()

    test_num = 10
    # test_num = len(bid_ids)
    error_idxes = {}

    for idx, bid_id in tqdm(enumerate(bid_ids[:test_num]), total=test_num):
        try:
            title = merge_infos(bid_id)
            merged_file = merge_2_json(bid_id)
            refined_file = refine_2_md(bid_id)
        except Exception as e:
            error_idxes[bid_id] = str(e)    # noqa

    print(error_idxes)


