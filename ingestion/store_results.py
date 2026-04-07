"""
배치 분석 결과 → ChromaDB "troublelens_results" 컬렉션 저장
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rich.console import Console

from config.settings import settings

console = Console()


# ── 1. 결과 컬렉션 로드 (없으면 자동 생성) ───────────────────────
def load_result_vectorstore() -> Chroma:
    return Chroma(
        collection_name="troublelens_results",  # ← 새 컬렉션
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        ),
        persist_directory="./data/chroma",
    )


# ── 2. 결과 JSON 로드 ─────────────────────────────────────────────
def load_batch_results(json_path: str | Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)
    console.print(f"[green]  결과 로드:[/] {len(results)}건")
    return results


# ── 3. ChromaDB 저장 형태로 변환 ──────────────────────────────────
def build_chroma_inputs(results: list[dict]) -> tuple[list, list, list]:
    """
    results → (documents, metadatas, ids) 변환

    documents : problem 필드 (임베딩 대상)
    metadatas : 나머지 필드 (검색 후 꺼내 쓸 부가정보)
    ids       : conversation_id (고유 식별자)

    ChromaDB metadatas 규칙:
       - list 타입 저장 불가 → json.dumps()로 문자열 변환 필수
       - None 저장 불가       → "" 빈 문자열로 대체
    """
    documents = []
    metadatas = []
    ids = []
    skipped = 0

    def safe_str(value) -> str:
        """str/int/float/bool 이외 타입은 전부 json.dumps로 문자열화"""
        if isinstance(value, (str, int, float, bool)):
            return value if isinstance(value, str) else str(value)
        return json.dumps(value, ensure_ascii=False)

    for item in results:
        if not item.get("problem"):
            console.print(
                f"[yellow]  건너뜀 (problem 없음):[/] {item['conversation_id'][:8]}..."
            )
            skipped += 1
            continue

        documents.append(item["problem"])

        metadatas.append(
            {
                "conversation_id": safe_str(item["conversation_id"]),
                "conversation_title": safe_str(item["conversation_title"]),
                "cause": safe_str(item["cause"] or ""),
                "solution": safe_str(item["solution"] or ""),
                "code_snippet": safe_str(item["code_snippet"] or ""),
                "tags": json.dumps(item["tags"], ensure_ascii=False),
                "category": safe_str(item["category"] or ""),
                "is_duplicate": str(item["is_duplicate"]),
                "analyzed_at": safe_str(item["analyzed_at"]),
                "chunk_text": safe_str(item.get("chunk_text", "")),
            }
        )

        ids.append(item["conversation_id"])

    return documents, metadatas, ids


# ── 4. ChromaDB에 저장 ────────────────────────────────────────────
def store_to_chroma(
    vectorstore: Chroma,
    documents: list,
    metadatas: list,
    ids: list,
) -> None:
    # 기존 id 삭제 후 추가 → upsert
    existing = vectorstore.get(ids=ids)
    existing_ids = existing["ids"]

    if existing_ids:
        vectorstore.delete(ids=existing_ids)
        console.print(f"[yellow]  기존 항목 삭제:[/] {len(existing_ids)}건 (덮어쓰기)")

    # LangChain 래퍼 메서드 사용 → embedding_function 자동 적용
    vectorstore.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids,
    )
    console.print(
        f"[bold green]  ChromaDB 저장 완료:[/] {len(ids)}건 → troublelens_results"
    )


# ── 5. 저장 결과 검증 ─────────────────────────────────────────────
def verify_store(vectorstore: Chroma) -> None:
    count = vectorstore._collection.count()
    console.print(f"[cyan]  컬렉션 총 항목 수:[/] {count}건")

    # 자연어 검색 테스트 1건
    test_query = "환경 설정 오류"
    search_results = vectorstore.similarity_search(test_query, k=1)

    if search_results:
        doc = search_results[0]
        console.print(f"\n[bold]검색 테스트:[/] '{test_query}'")
        console.print(f"  → 문제: {doc.page_content}")
        console.print(f"  → 해결: {doc.metadata['solution'][:80]}...")
    else:
        console.print("[yellow]  검색 결과 없음[/]")


# ── 6. 메인 실행 ──────────────────────────────────────────────────
def main(json_path: str | Path | None = None) -> None:
    console.print("[bold cyan]\n=== TroubleLens 결과 저장 시작 ===\n[/]")

    # json_path 미지정 시 가장 최근 파일 자동 선택
    if json_path is None:
        result_files = sorted(Path("./data/results").glob("batch_results_*.json"))
        if not result_files:
            console.print(
                "[red]  저장된 배치 결과 없음. 먼저 batch_runner.py 실행 필요[/]"
            )
            return
        json_path = result_files[-1]
        console.print(f"[green]  자동 선택:[/] {json_path.name}")

    results = load_batch_results(json_path)
    vectorstore = load_result_vectorstore()
    documents, metadatas, ids = build_chroma_inputs(results)
    store_to_chroma(vectorstore, documents, metadatas, ids)
    verify_store(vectorstore)

    console.print("[bold cyan]\n=== 저장 완료 ===\n[/]")


if __name__ == "__main__":
    main()
