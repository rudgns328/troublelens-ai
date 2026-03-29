"""
임베딩 + VectorDB 저장 모듈
Document(청크) → 벡터 변환 → ChromaDB 저장
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from ingestion.chunker import Document
from config.settings import settings
from rich.console import Console
from rich.progress import track

console = Console()


# ── 1. 임베더 클래스 ─────────────────────────────────────────────
class ConversationEmbedder:
    """
    Document 리스트 → ChromaDB에 임베딩 저장

    왜 OpenAIEmbeddings?
    - text-embedding-3-small: 빠르고 저렴, 1536차원
    - 한국어 포함 다국어 지원
    - 추후 Ollama 로컬 임베딩과 RAGAS로 비교 예정
    """

    def __init__(
        self,
        collection_name: str = "troublelens",  # ChromaDB 컬렉션 이름
        persist_dir: str = "./data/chroma",  # 로컬 저장 경로
    ):
        # OpenAI 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )

        # ChromaDB 초기화 (로컬 파일로 저장)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        console.print(f"[bold green] ChromaDB 초기화:[/] {persist_dir}")

    def embed_and_store(self, documents: list[Document]) -> None:
        """
        Document 리스트 → 임베딩 → ChromaDB 저장

        Chroma는 내부적으로 아래 3가지를 묶어서 저장
        - content  → 임베딩 벡터로 변환해서 저장
        - metadata → 그대로 저장 (검색 결과에 함께 반환)
        - id       → 중복 방지용 고유 키
        """
        console.print(f"[yellow]  임베딩 시작:[/] {len(documents)}개 청크")

        # Document → Chroma가 받는 형태로 변환
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [
            f"{doc.metadata['conversation_id']}_chunk_{doc.metadata['chunk_index']}"
            for doc in documents
        ]

        # 배치 단위로 나눠서 저장 (API 한 번에 너무 많이 보내면 오류)
        batch_size = 100
        for i in track(
            range(0, len(texts), batch_size),
            description="임베딩 저장 중...",
        ):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            self.vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

        console.print(f"[bold green] 저장 완료:[/] {len(documents)}개 청크 → ChromaDB")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        자연어 질문 → 유사 청크 검색

        k: 상위 몇 개 반환할지
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,  # 거리값 (낮을수록 유사)
            }
            for doc, score in results
        ]


# ── 2. 동작 확인 ─────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion.parser import ClaudeExportParser
    from ingestion.chunker import ConversationChunker

    # 파싱 + 청킹
    parser = ClaudeExportParser()
    conversations = parser.parse()
    candidates = parser.filter_candidates(conversations)

    chunker = ConversationChunker()
    all_docs = chunker.chunk_all(candidates)

    # 임베딩 + 저장
    embedder = ConversationEmbedder()
    embedder.embed_and_store(all_docs)

    # 검색 테스트
    console.print("\n[bold cyan] 검색 테스트[/]")
    results = embedder.search("Docker 네트워크 오류 해결 방법")

    for i, r in enumerate(results):
        console.print(f"\n[yellow]--- 결과 {i+1} (score: {r['score']:.4f}) ---[/]")
        console.print(f"출처: {r['metadata']['conversation_title']}")
        console.print(f"내용: {r['content'][:200]}")
