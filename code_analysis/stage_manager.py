from typing import Dict, Any

class ImprovedRFPStageManager:
    """
    RFP 작성 단계를 관리하는 클래스
    각 단계별 제목, 설명, 초기 질문 등을 관리합니다.
    """
    def __init__(self):
        self.stages: Dict[int, Dict[str, Any]] = {
            1: {
                "title": "기본 정보 수집",
                "description": "프로젝트 기본 정보 파악 후 초안 작성",
                "initial_question": (
                    "구축하고자 하는 시스템이나 서비스에 대해 간단히 설명해주세요.\n\n"
                    "예시: '공무원 업무매뉴얼 챗봇 구축', 'ERP 시스템 고도화', '모바일 앱 개발' 등"
                )
            },
            2: {
                "title": "사업개요",
                "description": "사업개요 초안 작성 및 검토",
                "initial_question": (
                    "1단계 정보를 바탕으로 사업개요를 작성하겠습니다.\n\n"
                    "특별히 강조할 배경/범위가 있으면 알려주세요. 없으면 '표준으로 작성'이라고 해주세요."
                )
            },
            3: {
                "title": "시스템 현황",
                "description": "현재 시스템 현황 초안 작성",
                "initial_question": (
                    "관련 시스템/인프라 현황을 아는 만큼 적어주세요.\n\n"
                    "모르면 '일반적인 수준으로 작성'이라고 해주세요."
                )
            },
            4: {
                "title": "사업추진방안",
                "description": "사업 추진방안 초안 작성",
                "initial_question": (
                    "추진 방향이나 일정에 대한 특별 요구가 있나요?\n\n"
                    "없다면 '표준적인 방식으로'라고 해주세요."
                )
            },
            5: {
                "title": "제안요청내용",
                "description": "제안요청 + 요구사항(이괄표/상세표)",
                "initial_question": (
                    "필요한 주요 기능들을 알려주세요.\n\n"
                    "예: '로그인, 파일 업로드, 검색, 관리자 대시보드, 통계 분석'\n\n"
                    "일반적인 요구사항은 '표준 요구사항으로'라고 해주세요."
                )
            }
        }

    def get_stage_info(self, stage: int) -> Dict[str, Any]:
        """
        주어진 단계의 정보를 반환합니다.
        
        Args:
            stage (int): 단계 번호 (1~5)
            
        Returns:
            Dict[str, Any]: 단계 정보 (제목, 설명, 초기 질문 등)
        """
        return self.stages.get(stage, self.stages[1])

    def get_initial_question(self, stage: int) -> str:
        """
        주어진 단계의 초기 질문을 반환합니다.
        
        Args:
            stage (int): 단계 번호 (1~5)
            
        Returns:
            str: 해당 단계의 초기 질문
        """
        return self.get_stage_info(stage)["initial_question"]
