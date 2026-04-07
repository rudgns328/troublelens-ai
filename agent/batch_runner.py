"""
배치 파이프라인 — ChromaDB 전체 청크 → LangGraph 분석 → 결과 저장
"""

import json
from pathlib import Path
from datetime import datetime

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.progress import track

from agent.graph import troublelens_graph
from agent.state import make_initial_state
from config.settings import settings

console = Console()


# ── 1. ChromaDB 로드 ──────────────────────────────────────────────
def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name="troublelens",
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        ),
        persist_directory="./data/chroma",
    )


# ── 2. 전체 청크 꺼내기 ───────────────────────────────────────────
def get_all_chunks(vectorstore: Chroma) -> dict:
    """
    ChromaDB에서 4203개 청크 전부 꺼내기
    반환 형태: {"ids": [...], "documents": [...], "metadatas": [...]}
    """
    result = vectorstore.get(
        include=["documents", "metadatas"]
        # embeddings는 제외 — 배치 처리엔 텍스트만 필요
    )
    console.print(f"[green]  전체 청크 수:[/] {len(result['ids'])}개")
    return result


# ── 3. conversation_id 기준으로 그룹핑 ───────────────────────────
def group_by_conversation(chunks: dict) -> dict[str, dict]:
    """
    청크를 대화 단위로 묶기

    반환 형태:
    {
      "conv_uuid_1": {
        "title": "SSH 오류 해결",
        "metadata": {...},          ← 대표 메타데이터 (chunk_index=0 기준)
        "chunks": [
          {"index": 0, "text": "..."},
          {"index": 1, "text": "..."},
        ]
      },
      ...
    }
    """
    grouped = {}

    for id_, text, meta in zip(
        chunks["ids"],
        chunks["documents"],
        chunks["metadatas"],
    ):
        conv_id = meta["conversation_id"]

        if conv_id not in grouped:
            grouped[conv_id] = {
                "title": meta["conversation_title"],
                "metadata": meta,
                "chunks": [],
            }

        grouped[conv_id]["chunks"].append(
            {
                "index": meta["chunk_index"],
                "text": text,
            }
        )

    # chunk_index 순서대로 정렬 (청크 순서 보장)
    for conv_id in grouped:
        grouped[conv_id]["chunks"].sort(key=lambda x: x["index"])

    console.print(f"[green]  대화 수:[/] {len(grouped)}개")
    return grouped


# ── 4. 대화별 배치 실행 ───────────────────────────────────────────
def run_batch(grouped: dict[str, dict]) -> list[dict]:
    """
    대화 단위로 LangGraph 실행

    is_troubleshooting == True인 결과만 수집해서 반환
    """
    results = []
    skipped = 0

    for conv_id, conv_data in track(
        grouped.items(),
        description="LangGraph 분석 중...",
    ):
        # 청크 텍스트 합치기 (순서 보장된 상태)
        full_text = "\n\n".join(c["text"] for c in conv_data["chunks"])

        # State 초기화 — chunk_id 자리에 conversation_id 사용
        state = make_initial_state(
            chunk_id=conv_id,
            chunk_text=full_text,
        )

        try:
            result = troublelens_graph.invoke(state)
        except Exception as e:
            console.print(f"[red]  오류 (건너뜀):[/] {conv_id[:8]}... → {e}")
            skipped += 1
            continue

        # 트러블슈팅이 아닌 대화는 결과에서 제외
        if not result["is_troubleshooting"] or not result.get("problem"):
            skipped += 1
            continue

        # 결과 저장 형태 정리
        results.append(
            {
                "conversation_id": conv_id,
                "conversation_title": conv_data["title"],
                "problem": result["problem"],
                "cause": result["cause"],
                "solution": result["solution"],
                "code_snippet": result["code_snippet"],
                "tags": result["tags"],
                "category": result["category"],
                "is_duplicate": result["is_duplicate"],
                "analyzed_at": datetime.now().isoformat(),
                "chunk_text": full_text,
            }
        )

    console.print(f"[bold green]  트러블슈팅 감지:[/] {len(results)}건")
    console.print(f"[yellow]  제외 (비-트러블슈팅/오류):[/] {skipped}건")
    return results


# ── 5. 결과 저장 ──────────────────────────────────────────────────
def save_results(results: list[dict]) -> Path:
    output_dir = Path("./data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"batch_results_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    console.print(f"[bold green]  저장 완료:[/] {output_path}")
    return output_path


# ── 6. 메인 실행 ──────────────────────────────────────────────────
def main():
    console.print("[bold cyan]\n=== TroubleㅋLens 배치 파이프라인 시작 ===\n[/]")

    vectorstore = load_vectorstore()
    chunks = get_all_chunks(vectorstore)
    grouped = group_by_conversation(chunks)
    results = run_batch(grouped)
    save_results(results)

    console.print("[bold cyan]\n=== 배치 완료 ===\n[/]")


if __name__ == "__main__":
    main()
