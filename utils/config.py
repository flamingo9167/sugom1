"""설정 로딩: Streamlit Secrets(클라우드) → 환경변수/.env(로컬) 순서로 읽는다."""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"


def get_config(key: str, default: str | None = None) -> str | None:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # secrets.toml 이 없는 로컬 환경
        pass
    value = os.getenv(key)
    return value if value else default


def get_api_key() -> str | None:
    return get_config("OPENAI_API_KEY")


def get_app_password() -> str | None:
    """설정하면 앱 사용 전에 비밀번호를 묻는다 (공개 배포 시 API 비용 남용 방지)."""
    return get_config("APP_PASSWORD")


def get_model() -> str:
    return get_config("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
