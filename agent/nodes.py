import re
import json
import logging
from langchain.schema import HumanMessage, SystemMessage
from agent.state import TroubleshootingState
from dotenv import load_dotenv
from llm import BaseLLM

load_dotenv()

logger = logging.getLogger(__name__)


# ── JSON 파싱 유틸 ────────────────────────────────────────────
def parse_json_response(text: str, node: str) -> dict | None:
    """
    LLM 응답에서 JSON 안전하게 추출

    처리 순서:
      1. 마크다운 펜스 제거 (```json ... ```)
      2. 순수 JSON 파싱 시도
      3. { } 블록 추출 후 재시도
      4. 역슬래시 수리 후 재시도
      5. 실패 시 None 반환
    """
    text = text.strip()

    # 1단계: 마크다운 코드펜스 제거
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2단계: 순수 JSON 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3단계: { } 블록 추출
    brace_match = re.search(r"\{[\s\S]*\}", text)
    candidate = brace_match.group() if brace_match else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 4단계: 역슬래시 수리
    # Windows 경로(C:\Users\...)처럼 JSON 비표준 이스케이프 처리
    # \u, \n, \t, \r, \", \\, \/ 외의 역슬래시를 \\로 치환
    try:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning("JSON 파싱 실패 | node: %s | 응답: %s", node, text[:200])
    return None


# ── Node 1: 트러블슈팅 판별 ───────────────────────────────────
def detect_troubleshooting(state: TroubleshootingState, llm: BaseLLM) -> dict:
    """이 청크가 트러블슈팅 내용인지 판별한다."""

    messages = [
        SystemMessage(
            content="""당신은 개발 대화를 분석하는 전문가입니다.
주어진 텍스트가 트러블슈팅(오류 해결, 문제 디버깅, 설정 문제 등)을 담고 있는지 판단하세요.

반드시 아래 JSON 형식으로만 답하세요. 마크다운, 코드블록, 설명 텍스트 없이 JSON만 출력하세요:
{"is_troubleshooting": true}"""
        ),
        HumanMessage(content=f"다음 텍스트를 분석하세요:\n\n{state['chunk_text']}"),
    ]

    response = llm.invoke(messages)
    result = parse_json_response(response, node="detect")

    if result is None:
        return {"is_troubleshooting": False}

    return {"is_troubleshooting": result.get("is_troubleshooting", False)}


# ── Node 2: 정보 추출 ─────────────────────────────────────────
def extract_information(state: TroubleshootingState, llm: BaseLLM) -> dict:
    """트러블슈팅에서 문제/원인/해결책/코드를 추출한다."""

    messages = [
        SystemMessage(
            content="""개발 트러블슈팅 텍스트에서 정보를 추출하세요.

반드시 아래 JSON 형식으로만 답하세요. 마크다운, 코드블록, 설명 텍스트 없이 JSON만 출력하세요:
{
  "problem": "문제 상황 한 문장 요약",
  "cause": "원인 한 문장 요약",
  "solution": "해결책 한 문장 요약",
  "code_snippet": "핵심 코드 한 줄 또는 명령어 문자열. 코드가 여러 줄이면 줄바꿈(\\n)으로 연결한 하나의 문자열로. 없으면 null"
}"""
        ),
        HumanMessage(content=f"다음 트러블슈팅을 분석하세요:\n\n{state['chunk_text']}"),
    ]

    response = llm.invoke(messages)
    result = parse_json_response(response, node="extract")

    if result is None:
        return {"problem": None, "cause": None, "solution": None, "code_snippet": None}

    return {
        "problem": result.get("problem"),
        "cause": result.get("cause"),
        "solution": result.get("solution"),
        "code_snippet": result.get("code_snippet"),
    }


# ── Node 3: 태깅 & 분류 ──────────────────────────────────────
def tag_and_classify(state: TroubleshootingState, llm: BaseLLM) -> dict:
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

반드시 아래 JSON 형식으로만 답하세요. 마크다운, 코드블록, 설명 텍스트 없이 JSON만 출력하세요:
{
  "tags": [],
  "category": ""
}"""
        ),
        HumanMessage(content=f"다음 트러블슈팅을 분류하세요:\n\n{state['chunk_text']}"),
    ]

    response = llm.invoke(messages)
    result = parse_json_response(response, node="tag")

    if result is None:
        return {"tags": [], "category": "기타"}

    return {
        "tags": result.get("tags", []),
        "category": result.get("category", "기타"),
    }


# ── Node 4: 중복 감지 ─────────────────────────────────────────
def detect_duplicate(state: TroubleshootingState) -> dict:
    """일단 중복 없음으로 기본 처리 (Phase 4에서 ChromaDB 검색으로 고도화 예정)"""
    return {
        "is_duplicate": False,
        "duplicate_of": None,
    }
