from agent.graph import troublelens_graph
from agent.state import make_initial_state

input_state = make_initial_state(
    chunk_id="test-001",
    chunk_text="""
    httpx 버전 업데이트 후 'proxies' 키워드 오류가 발생했습니다.
    langchain이 내부적으로 httpx를 사용하는데, 최신 버전에서
    proxies 파라미터가 제거되었기 때문입니다.
    requirements.txt에 httpx==0.27.0을 고정해서 해결했습니다.
    """,
)

# 비-트러블슈팅 케이스
# input_state = make_initial_state(
#     chunk_id="test-002",
#     chunk_text="""
#     LangGraph는 LLM 기반 애플리케이션을 위한 상태 기반 그래프 프레임워크입니다.
#     노드와 엣지로 구성되며, 복잡한 워크플로우를 표현할 수 있습니다.
#     """,
# )

result = troublelens_graph.invoke(input_state)

print("=== 분석 결과 ===")
print(f"트러블슈팅 여부: {result['is_troubleshooting']}")
print(f"문제:    {result['problem']}")
print(f"원인:    {result['cause']}")
print(f"해결책:  {result['solution']}")
print(f"태그:    {result['tags']}")
print(f"카테고리: {result['category']}")
