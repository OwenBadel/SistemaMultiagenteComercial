"""
Configuracion del puente API entre Botpress, Supabase PostgreSQL,
Google Calendar API v3, Ollama (Meta Llama local) y SistemaMultiagenteComercial.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _leer_bool_env(nombre: str, default: bool = True) -> bool:
    valor = os.getenv(nombre)
    if valor is None:
        return default
    return valor.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class ConfiguracionBotpress:
    """Parametros del servidor HTTP conectado con Botpress, Supabase, Google Calendar y Ollama."""

    botpress_api_key: str = ""
    human_reactivation_api_key: str = ""
    owner_whatsapp_number: str = ""
    supabase_url: str = ""
    supabase_key: str = ""
    negocio_id: str = "tech_freelance"
    sqlite_db_path: str = "data/entorno_local.db"
    google_calendar_id: str = ""
    google_service_account_json: str = ""
    google_timezone: str = "America/Bogota"
    ollama_host: str = "http://127.0.0.1:11434"
    llama_model: str = "auto"
    llama_timeout_seconds: float = 30.0
    llama_auto_install: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def desde_entorno(cls, ruta_env: Path | None = None) -> "ConfiguracionBotpress":
        """Carga variables desde `.env` y el entorno del sistema."""
        if ruta_env is not None and ruta_env.exists():
            load_dotenv(dotenv_path=ruta_env, override=False)
        else:
            load_dotenv(override=False)

        return cls(
            botpress_api_key=os.getenv("BOTPRESS_API_KEY", ""),
            human_reactivation_api_key=os.getenv(
                "HUMAN_REACTIVATION_API_KEY", ""
            ),
            owner_whatsapp_number=os.getenv("OWNER_WHATSAPP_NUMBER", ""),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
            negocio_id=os.getenv("NEGOCIO_ID", "tech_freelance"),
            sqlite_db_path=os.getenv("SQLITE_DB_PATH", "data/entorno_local.db"),
            google_calendar_id=os.getenv("GOOGLE_CALENDAR_ID", ""),
            google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", ""),
            google_timezone=os.getenv("GOOGLE_TIMEZONE", "America/Bogota"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            llama_model=os.getenv("LLAMA_MODEL", "auto"),
            llama_timeout_seconds=float(os.getenv("LLAMA_TIMEOUT_SECONDS", "30.0")),
            llama_auto_install=_leer_bool_env("LLAMA_AUTO_INSTALL", True),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )

    @property
    def usa_supabase(self) -> bool:
        """Indica si hay credenciales listas para conectar con Supabase Cloud."""
        return bool(self.supabase_url.strip() and self.supabase_key.strip())

    @property
    def usa_google_calendar(self) -> bool:
        """Indica si hay credenciales listas para Google Calendar API v3."""
        return bool(
            self.google_calendar_id.strip() and self.google_service_account_json.strip()
        )

    @property
    def google_calendar_configurado(self) -> bool:
        """Indica que se intento configurar Calendar, incluso con datos incompletos."""
        return bool(
            self.google_calendar_id.strip()
            or self.google_service_account_json.strip()
        )
