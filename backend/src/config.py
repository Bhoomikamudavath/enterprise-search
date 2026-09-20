from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SEARCH_")

    api_key: str = "dev-local-key-change-me"
    allowed_origins: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    rate_limit: str = "30/minute"
    log_level: str = "INFO"
    model_name: str = "all-MiniLM-L6-v2"
    data_dir: str = "data"


settings = Settings()