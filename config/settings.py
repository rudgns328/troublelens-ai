from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # 경로 설정
    raw_data_dir: Path = Path("./data/raw")
    chroma_persist_dir: Path = Path("./data/chroma")

    # LLM 설정
    llm_model: str = "gpt-4o-mini"  # 비용 절약용 기본값
    embedding_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"


settings = Settings()
