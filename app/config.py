from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # RGS_MODE: str = "app"
    WEIGHTS: List[int] = [1, 1, 1]
    ALPHA: float = 0.5
    N: int = 12
    DAYS: int = 7
    PROTOCOLS_PER_DAY: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # or "allow" if you prefer
    )