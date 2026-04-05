from langgraph.graph import StateGraph, END
from agent.state import TroubleshootingState
from agent.nodes import (
    detect_troubleshooting,
    extract_information,
    tag_and_classify,
    detect_duplicate,
)


def route_after_detection(state: TroubleshootingState) -> str:
    """Node 1 이후 분기: 트러블슈팅이면 추출로, 아니면 종료."""
    if state["is_troubleshooting"]:
        return "extract"
    return END


def build_graph():
    graph = StateGraph(TroubleshootingState)

    graph.add_node("detect", detect_troubleshooting)
    graph.add_node("extract", extract_information)
    graph.add_node("tag", tag_and_classify)
    graph.add_node("duplicate", detect_duplicate)

    graph.set_entry_point("detect")

    # 조건부 Edge (Node 1 이후 분기)
    graph.add_conditional_edges(
        "detect", route_after_detection, {"extract": "extract", END: END}
    )

    graph.add_edge("extract", "tag")
    graph.add_edge("tag", "duplicate")
    graph.add_edge("duplicate", END)

    return graph.compile()


# 그래프 인스턴스 (다른 파일에서 import해서 쓸 수 있도록)
troublelens_graph = build_graph()
