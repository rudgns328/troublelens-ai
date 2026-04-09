from abc import ABC, abstractmethod
from langchain.schema import BaseMessage


class BaseLLM(ABC):
    """
    LLM 추상 인터페이스
    """

    @abstractmethod
    def invoke(self, messages: list[BaseMessage]) -> str:
        """
        메시지 리스트를 받아 LLM 응답 문자열을 반환한다.
        반환 타입이 str인 이유: nodes.py의 parse_json_response가
        문자열을 받도록 설계되어 있기 때문
        """
        pass
