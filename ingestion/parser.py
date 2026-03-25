"""
Claude Export JSON 파서
conversations.json을 파싱해서 구조화된 대화 데이터로 변환
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

console = Console()


# ── 1. 데이터 모델 정의 ──────────────────────────────────────────
@dataclass
class Message:
    """대화 메시지 하나를 표현하는 모델"""

    uuid: str
    sender: str  # "human" or "assistant"
    text: str  # thinking 블록 제외한 순수 텍스트
    created_at: datetime


@dataclass
class Conversation:
    """대화 전체를 표현하는 모델"""

    uuid: str
    name: str
    summary: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def full_text(self) -> str:
        """전체 대화를 하나의 텍스트로 합치기 (RAG 청킹용)"""
        parts = []
        for msg in self.messages:
            role = "사용자" if msg.sender == "human" else "Claude"
            parts.append(f"[{role}]\n{msg.text}")
        return "\n\n".join(parts)


# ── 2. 핵심 파싱 함수 ────────────────────────────────────────────
def _extract_clean_text(content_blocks: list[dict]) -> str:
    """
    content 배열에서 순수 텍스트만 추출
    - type == "thinking" 블록은 제외 (Claude 내부 사고 과정)
    - type == "text" 블록만 합쳐서 반환
    """
    text_parts = []
    for block in content_blocks:
        if block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def _parse_message(raw_msg: dict) -> Optional[Message]:
    """메시지 딕셔너리 → Message 객체 변환"""
    content_blocks = raw_msg.get("content", [])
    clean_text = _extract_clean_text(content_blocks)

    # 텍스트가 없는 메시지는 스킵 (예: 파일만 첨부된 메시지)
    if not clean_text:
        return None

    return Message(
        uuid=raw_msg["uuid"],
        sender=raw_msg["sender"],
        text=clean_text,
        created_at=datetime.fromisoformat(raw_msg["created_at"].replace("Z", "+00:00")),
    )


def _parse_conversation(raw_conv: dict) -> Conversation:
    """대화 딕셔너리 → Conversation 객체 변환"""
    messages = []
    for raw_msg in raw_conv.get("chat_messages", []):
        msg = _parse_message(raw_msg)
        if msg:
            messages.append(msg)

    return Conversation(
        uuid=raw_conv["uuid"],
        name=raw_conv.get("name", "제목 없음"),
        summary=raw_conv.get("summary", ""),
        created_at=datetime.fromisoformat(
            raw_conv["created_at"].replace("Z", "+00:00")
        ),
        updated_at=datetime.fromisoformat(
            raw_conv["updated_at"].replace("Z", "+00:00")
        ),
        messages=messages,
    )


# ── 3. 메인 파서 클래스 ──────────────────────────────────────────
class ClaudeExportParser:
    """Claude Export JSON 파일을 파싱하는 메인 클래스"""

    def __init__(self, raw_data_dir: str | Path = "./data/raw"):
        self.raw_data_dir = Path(raw_data_dir)

    def parse(self, filename: str = "conversations.json") -> list[Conversation]:
        """JSON 파일 → Conversation 리스트 반환"""
        filepath = self.raw_data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없어요: {filepath}")

        console.print(f"[bold green]📂 파싱 시작:[/] {filepath}")

        with open(filepath, encoding="utf-8") as f:
            raw_data = json.load(f)

        conversations = [_parse_conversation(raw) for raw in raw_data]

        console.print(f"[bold green]✅ 파싱 완료:[/] {len(conversations)}개 대화")
        return conversations

    def filter_candidates(
        self,
        conversations: list[Conversation],
        min_messages: int = 2,
        min_text_length: int = 50,
    ) -> list[Conversation]:
        """
        1차 필터: LLM 판별 전 명백히 의미 없는 대화 제거

        기준:
        - 메시지 수 2개 미만 제거 (질문만 하고 끝)
        - 전체 텍스트 50자 미만 제거 ("안녕" 수준)
        - 제목 없는 것 제거
        """
        before = len(conversations)

        filtered = [
            conv
            for conv in conversations
            if conv.message_count >= min_messages
            and len(conv.full_text) >= min_text_length
            and conv.name.strip()
        ]

        after = len(filtered)
        console.print(
            f"[yellow]🔍 1차 필터:[/] {before}개 → {after}개 "
            f"([red]{before - after}개 제거[/])"
        )
        return filtered


# ── 4. 간단한 동작 확인용 ────────────────────────────────────────
if __name__ == "__main__":
    parser = ClaudeExportParser()

    # 파싱
    conversations = parser.parse()

    # 1차 필터
    candidates = parser.filter_candidates(conversations)

    print(f"\n✅ LLM 판별 대상: {len(candidates)}개")
    for c in candidates:
        print(f"  [{c.message_count:3}턴] {c.name}")
