"""
청킹 전략 모듈
Conversation 객체 → Document(청크) 리스트로 변환
"""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.parser import Conversation


# ── 1. 청크 데이터 모델 ──────────────────────────────────────────
@dataclass
class Document:
    """
    VectorDB에 저장될 하나의 청크 단위

    content   : 실제 텍스트 (임베딩 대상)
    metadata  : 검색 결과에 함께 반환될 부가 정보
    """

    content: str
    metadata: dict


# ── 2. 청커 클래스 ───────────────────────────────────────────────
class ConversationChunker:
    """
    Conversation → Document 리스트로 변환하는 청커

    왜 RecursiveCharacterTextSplitter?
    - 문단(\n\n) → 문장(\n) → 단어 순서로 재귀적으로 자름
    - 가능하면 의미 단위(문단)를 유지하려고 시도함
    - 트러블슈팅 대화처럼 문맥이 중요한 텍스트에 적합
    """

    def __init__(
        self,
        chunk_size: int = 500,  # 한 청크의 최대 글자 수
        chunk_overlap: int = 50,  # 청크 간 겹치는 글자 수
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 한국어 대화 특성상 문단 → 줄바꿈 → 문장 순으로 자름
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def chunk(self, conversation: Conversation) -> list[Document]:
        """
        Conversation 1개 → Document 리스트 반환
        """
        full_text = conversation.full_text

        # RecursiveCharacterTextSplitter로 텍스트 분할
        chunks = self.splitter.split_text(full_text)

        documents = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                content=chunk_text,
                metadata={
                    "conversation_id": conversation.uuid,
                    "conversation_title": conversation.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                    "message_count": conversation.message_count,
                },
            )
            documents.append(doc)

        return documents

    def chunk_all(self, conversations: list[Conversation]) -> list[Document]:
        """Conversation 리스트 전체를 청킹"""
        all_docs = []
        for conv in conversations:
            docs = self.chunk(conv)
            all_docs.extend(docs)
        return all_docs


if __name__ == "__main__":
    from ingestion.parser import ClaudeExportParser

    parser = ClaudeExportParser()
    conversations = parser.parse()
    candidates = parser.filter_candidates(conversations)

    chunker = ConversationChunker()
    all_docs = chunker.chunk_all(candidates)

    print(f"총 청크 수: {len(all_docs)}")
    print(f"\n--- 첫 번째 청크 확인 ---")
    print(f"내용: {all_docs[0].content[:200]}")
    print(f"메타데이터: {all_docs[0].metadata}")
