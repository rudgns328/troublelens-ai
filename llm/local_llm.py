import logging
from langchain_ollama import ChatOllama
from langchain.schema import BaseMessage
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama 로컬 모델 기반 LLM 구현체."""

    def __init__(self, model: str = "exaone:latest", temperature: float = 0):
        self._llm = ChatOllama(model=model, temperature=temperature)
        logger.info("OllamaLLM 초기화 완료 | model=%s", model)

    def invoke(self, messages: list[BaseMessage]) -> str:
        response = self._llm.invoke(messages)
        return response.content
