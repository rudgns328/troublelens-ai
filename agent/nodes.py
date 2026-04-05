import json
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from agent.state import TroubleshootingState
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Node 1: 트러블슈팅 판별 ───────────────────────────────────
def detect_troubleshooting(state: TroubleshootingState) -> dict:
    """이 청크가 트러블슈팅 내용인지 판별한다."""

    messages = [
        SystemMessage(
            content="""당신은 개발 대화를 분석하는 전문가입니다.
주어진 텍스트가 트러블슈팅(오류 해결, 문제 디버깅, 설정 문제 등)을 담고 있는지 판단하세요.

반드시 JSON으로만 답하세요. is_troubleshooting 값은 반드시 true 또는 false:
{"is_troubleshooting": true}"""
        ),
        HumanMessage(content=f"다음 텍스트를 분석하세요:\n\n{state['chunk_text']}"),
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        return {"is_troubleshooting": result.get("is_troubleshooting", False)}
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패 | node: detect | 응답: %s", response.content)
        return {"is_troubleshooting": False}


# ── Node 2: 정보 추출 ─────────────────────────────────────────
def extract_information(state: TroubleshootingState) -> dict:
    """트러블슈팅에서 문제/원인/해결책/코드를 추출한다."""

    messages = [
        SystemMessage(
            content="""개발 트러블슈팅 텍스트에서 정보를 추출하세요.

반드시 JSON으로만 답하세요:
{
  "problem": "문제 상황 한 문장 요약",
  "cause": "원인 한 문장 요약",
  "solution": "해결책 한 문장 요약",
  "code_snippet": "핵심 코드 (없으면 null)"
}"""
        ),
        HumanMessage(content=f"다음 트러블슈팅을 분석하세요:\n\n{state['chunk_text']}"),
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        return {
            "problem": result.get("problem"),
            "cause": result.get("cause"),
            "solution": result.get("solution"),
            "code_snippet": result.get("code_snippet"),
        }
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패 | node: extract | 응답: %s", response.content)
        return {"problem": None, "cause": None, "solution": None, "code_snippet": None}


# ── Node 3: 태깅 & 분류 ──────────────────────────────────────
def tag_and_classify(state: TroubleshootingState) -> dict:
    """기술 스택 태그와 카테고리를 분류한다."""

    messages = [
        SystemMessage(
            content="""개발 트러블슈팅 텍스트에 태그와 카테고리를 붙이세요.

카테고리는 다음 중 하나:
"설치/환경" | "버그" | "설계" | "성능" | "기타"

tags 규칙:
- 텍스트에 등장하는 기술 스택을 빠짐없이 추출하세요
- 언어, 라이브러리, 프레임워크, 툴, 에러 유형 모두 포함합니다
- 반드시 1개 이상 추출해야 합니다
- 기술 스택이 전혀 언급되지 않은 경우에만 빈 배열 []을 사용하세요

반드시 JSON으로만 답하세요:
{
  "tags": [],
  "category": ""
}"""
        ),
        HumanMessage(content=f"다음 트러블슈팅을 분류하세요:\n\n{state['chunk_text']}"),
    ]

    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)
        return {
            "tags": result.get("tags", []),
            "category": result.get("category", "기타"),
        }
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패 | node: tag | 응답: %s", response.content)
        return {"tags": [], "category": "기타"}


# ── Node 4: 중복 감지 ─────────────────────────────────────────
def detect_duplicate(state: TroubleshootingState) -> dict:
    """일단 중복 없음으로 기본 처리 (Phase 4에서 ChromaDB 검색으로 고도화 예정)"""
    return {
        "is_duplicate": False,
        "duplicate_of": None,
    }
