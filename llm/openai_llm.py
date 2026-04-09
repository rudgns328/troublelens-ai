import logging
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI 기반 LLM 구현체."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self._llm = ChatOpenAI(model=model, temperature=temperature)
        logger.info("OpenAILLM 초기화 완료 | model=%s", model)

    def invoke(self, messages: list[BaseMessage]) -> str:
        response = self._llm.invoke(messages)
        return response.content
