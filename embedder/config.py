"""
Centralised settings loader for the embedder worker.
Change defaults here; override with env-vars or a `.env` file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from common.models import encoder

class Settings(BaseSettings):
    images_dir: Path = Field("/data/images", env="IMAGES_DIR")
    es_host: str = Field("http://es:9200", env="ES_HOST")
    batch_size: int = Field(32, env="BATCH_SIZE")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    @property
    def vector_dim(self) -> int:
        return encoder.embed_dim
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
