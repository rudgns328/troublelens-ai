from agent.graph import build_graph

# GPT 테스트
print("=== GPT 테스트 ===")
graph = build_graph(llm_type="openai")
state = {"chunk_text": "pip install 중 httpx 버전 충돌 에러가 발생했다. httpx==0.27.0으로 고정해서 해결했다."}
result = graph.invoke(state)
print(result)

# Ollama 테스트
print("\n=== Ollama 테스트 ===")
graph = build_graph(llm_type="ollama")
result = graph.invoke(state)
print(result)