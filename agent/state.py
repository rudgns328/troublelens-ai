from typing import TypedDict, Optional


class TroubleshootingState(TypedDict):
    # ── 입력 ──────────────────────────────────
    chunk_id: str  # ChromaDB에 저장된 청크 고유 ID
    chunk_text: str  # 분석할 청크 텍스트 원문

    # ── Node 1: 트러블슈팅 판별 ────────────────
    is_troubleshooting: bool  # 트러블슈팅 여부

    # ── Node 2: 정보 추출 ──────────────────────
    problem: Optional[str]  # 문제 상황
    cause: Optional[str]  # 원인
    solution: Optional[str]  # 해결책
    code_snippet: Optional[str]  # 관련 코드 (있을 경우)

    # ── Node 3: 태깅 & 분류 ───────────────────
    tags: list[str]  # 기술 스택 태그
    category: Optional[str]  # 카테고리 ("설치/환경", "버그", "설계" 등)

    # ── Node 4: 중복 감지 ──────────────────────
    is_duplicate: bool  # 중복 여부
    duplicate_of: Optional[str]  # 중복이면 원본 chunk_id


def make_initial_state(chunk_id: str, chunk_text: str) -> TroubleshootingState:
    """ChromaDB 청크로부터 초기 State를 만든다."""
    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "is_troubleshooting": False,
        "problem": None,
        "cause": None,
        "solution": None,
        "code_snippet": None,
        "tags": [],
        "category": None,
        "is_duplicate": False,
        "duplicate_of": None,
    }
