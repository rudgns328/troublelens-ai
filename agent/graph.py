from functools import partial
from langgraph.graph import StateGraph, END
from agent.state import TroubleshootingState
from agent.nodes import (
    detect_troubleshooting,
    extract_information,
    tag_and_classify,
    detect_duplicate,
)
from llm import OpenAILLM, OllamaLLM


def route_after_detection(state: TroubleshootingState) -> str:
    """Node 1 이후 분기: 트러블슈팅이면 추출로, 아니면 종료."""
    if state["is_troubleshooting"]:
        return "extract"
    return END


def build_graph(llm_type: str = "openai"):
    # LLM 선택
    if llm_type == "openai":
        llm = OpenAILLM()
    else:
        llm = OllamaLLM()

    graph = StateGraph(TroubleshootingState)

    # partial로 llm 미리 고정
    graph.add_node("detect", partial(detect_troubleshooting, llm=llm))
    graph.add_node("extract", partial(extract_information, llm=llm))
    graph.add_node("tag", partial(tag_and_classify, llm=llm))
    graph.add_node("duplicate", detect_duplicate)

    graph.set_entry_point("detect")

    graph.add_conditional_edges(
        "detect", route_after_detection, {"extract": "extract", END: END}
    )

    graph.add_edge("extract", "tag")
    graph.add_edge("tag", "duplicate")
    graph.add_edge("duplicate", END)

    return graph.compile()


# 기본은 GPT
troublelens_graph = build_graph(llm_type="openai")
