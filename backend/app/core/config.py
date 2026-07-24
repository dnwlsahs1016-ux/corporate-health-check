from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # 서빙 전용 배포(Streamlit 등)에서는 이 값들이 없어도 되므로 기본값을 둔다.
    # 실제 DART/ECOS 수집 스크립트를 돌릴 때만 .env에 진짜 키가 필요하다.
    dart_api_key: str = ""
    ecos_api_key: str = ""

    start_year: int = 2015
    end_year: int = 2024


settings = Settings()

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
