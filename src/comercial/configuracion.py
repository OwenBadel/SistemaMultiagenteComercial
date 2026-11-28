"""Configuracion centralizada de la plataforma comercial multi-negocio."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

_PLACEHOLDERS_SESSION = {
    "",
    "cambia_este_secreto_largo_en_produccion",
    "admin_sma_2026",
}


def _env_bool(nombre: str, default: bool = False) -> bool:
    raw = os.getenv(nombre)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "si", "on"}


@dataclass
class ConfiguracionComercial:
    """Variables de entorno para el backend comercial multi-tenant."""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    supabase_url: str = ""
    supabase_key: str = ""
    sqlite_db_path: str = "data/comercial_multitenant.db"
    google_service_account_json: str = ""
    google_calendar_id_default: str = ""
    google_timezone: str = "America/Bogota"
    admin_access_key: str = "admin_sma_2026"
    session_secret: str = ""
    botpress_api_key: str = ""
    whatsapp_verify_token: str = "sma_whatsapp_verify_2026"
    whatsapp_app_secret: str = ""
    whatsapp_default_access_token: str = ""
    allow_insecure_webhooks: bool = False
    public_base_url: str = ""
    negocio_id_default: str = "estudio_fotografia"
    host: str = "0.0.0.0"
    port: int = 8000
    _session_secret_efimero: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        secreto = (self.session_secret or "").strip()
        if secreto in _PLACEHOLDERS_SESSION:
            self.session_secret = secrets.token_urlsafe(32)
            self._session_secret_efimero = True
        else:
            self.session_secret = secreto

    @classmethod
    def desde_entorno(cls, ruta_env: Path | None = None) -> "ConfiguracionComercial":
        if ruta_env is not None and ruta_env.exists():
            load_dotenv(dotenv_path=ruta_env, override=False)
        else:
            load_dotenv(override=False)

        admin_key = (
            os.getenv("ADMIN_ACCESS_KEY")
            or os.getenv("HUMAN_REACTIVATION_API_KEY")
            or "admin_sma_2026"
        ).strip()
        session_secret = os.getenv("SESSION_SECRET", "").strip()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite",
            supabase_url=os.getenv("SUPABASE_URL", "").strip(),
            supabase_key=os.getenv("SUPABASE_KEY", "").strip(),
            sqlite_db_path=os.getenv("SQLITE_DB_PATH", "data/comercial_multitenant.db").strip(),
            google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip(),
            google_calendar_id_default=os.getenv("GOOGLE_CALENDAR_ID", "").strip(),
            google_timezone=os.getenv("GOOGLE_TIMEZONE", "America/Bogota").strip() or "America/Bogota",
            admin_access_key=admin_key,
            session_secret=session_secret,
            botpress_api_key=os.getenv("BOTPRESS_API_KEY", "").strip(),
            whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN", "sma_whatsapp_verify_2026").strip(),
            whatsapp_app_secret=os.getenv("WHATSAPP_APP_SECRET", "").strip(),
            whatsapp_default_access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip(),
            allow_insecure_webhooks=_env_bool("ALLOW_INSECURE_WEBHOOKS", default=False),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
            negocio_id_default=os.getenv("NEGOCIO_ID", "estudio_fotografia").strip() or "estudio_fotografia",
            host=os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
        )

    @property
    def usa_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def usa_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    def advertencias_seguridad(self) -> List[str]:
        avisos: List[str] = []
        if self.admin_access_key == "admin_sma_2026":
            avisos.append("ADMIN_ACCESS_KEY usa el valor de desarrollo por defecto; cambialo en .env.")
        if self._session_secret_efimero:
            avisos.append(
                "SESSION_SECRET no esta definido en .env; se genero uno aleatorio efimero (las sesiones cerraran al reiniciar)."
            )
        if self.allow_insecure_webhooks:
            avisos.append(
                "ALLOW_INSECURE_WEBHOOKS=true esta activo; desactivalo en produccion y configura BOTPRESS_API_KEY / WHATSAPP_APP_SECRET."
            )
        elif not self.botpress_api_key and not self.whatsapp_app_secret:
            avisos.append(
                "BOTPRESS_API_KEY y WHATSAPP_APP_SECRET estan vacios; los webhooks externos responderan 503 hasta configurarlos."
            )
        return avisos
