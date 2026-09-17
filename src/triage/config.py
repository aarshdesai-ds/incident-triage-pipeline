import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved relative to this file, not the current working directory, so it
# works whether you run from src/triage/ (uvicorn, run_sample_batch.py) or
# the project root (pytest). This is also where GEMINI_API_KEY / GOOGLE_API_KEY
# actually reach os.environ for the SDK to find - Settings deliberately never
# models the credential itself (see Option A), so without this call it's never
# loaded from the file at all.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore", frozen=True)

    model: str = Field(default="gemini-2.5-flash")
    max_tokens: int = Field(default=256,ge=1,le=4096)
    request_timeout_s: float = Field(default=30.0, gt=0)
    sdk_max_retries: int = Field(default = 3, ge=0, le=10)
    app_max_attempts: int = Field(default=1, ge=1, le=10)
    batch_max_concurrency: int = Field(default=8, ge=1, le=50)
    # Reserved: Gemini has no direct equivalent of a discounted async batch
    # endpoint yet. batch.py currently always uses bounded live concurrency;
    # this flag is unused until that changes.
    batch_use_batches_api: bool = Field(default=False)
    database_url: str = Field(default="sqlite:///./triage.db")
    log_level: Literal["DEBUG","INFO","WARNING","ERROR"] = Field(default="INFO")
    log_json: bool = Field(default=True)
    environment: Literal["dev","staging","prod"] = Field(default="dev")
    service_name: str = Field(default="incident-triage")
    system_prompt_path: Path | None = None

    

    @field_validator("database_url")
    @classmethod
    def _non_empty_db_url(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("database_url must not be empty")
        return value.strip()

    @field_validator("system_prompt_path")
    @classmethod
    def _prompt_path_exists(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_file():
            raise ValueError(f"system_prompt_path does not exist: {value}")
        return value

    @model_validator(mode="after")
    def validate_prod(self) -> "Settings":
        if self.environment == "prod":
            if not self.log_json:
                raise ValueError("JSON Logging should be set to true for production environment")
            if self.database_url.startswith("sqlite"):
                raise ValueError("SQLite cannot be the database for the production environment")
        return self

    @model_validator(mode="after") 
    def validate_retries(self) -> "Settings":
        if self.app_max_attempts > 1 and self.sdk_max_retries > 3:
            warnings.warn("stacking SDK retries and app retries", stacklevel=2)
        return self
    
    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"
    
    def system_prompt_override(self) -> str | None:
        if self.system_prompt_path is None:
            return None
        return self.system_prompt_path.read_text(encoding="utf-8")


@lru_cache
def get_settings() -> "Settings":
    return Settings()
