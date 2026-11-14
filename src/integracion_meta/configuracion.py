"""
Configuracion de credenciales y entorno para la integracion con Meta Graph API
(WhatsApp Cloud API, Facebook Messenger e Instagram Direct).

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


@dataclass
class ConfiguracionMeta:
    """Parametros de conexion hacia Meta for Developers y servidor Webhook."""

    meta_verify_token: str = "sma_comercial_verify_2026"
    meta_app_secret: str = ""
    meta_api_version: str = "v21.0"
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    admin_whatsapp_number: str = ""
    messenger_page_access_token: str = ""
    instagram_access_token: str = ""
    instagram_account_id: str = ""
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def desde_entorno(cls, ruta_env: Path | None = None) -> "ConfiguracionMeta":
        """Carga las variables desde `.env` y el entorno del sistema."""
        if ruta_env is not None and ruta_env.exists():
            load_dotenv(dotenv_path=ruta_env, override=False)
        else:
            load_dotenv(override=False)

        return cls(
            meta_verify_token=os.getenv("META_VERIFY_TOKEN", "sma_comercial_verify_2026"),
            meta_app_secret=os.getenv("META_APP_SECRET", ""),
            meta_api_version=os.getenv("META_API_VERSION", "v21.0"),
            whatsapp_access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
            whatsapp_phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""),
            admin_whatsapp_number=os.getenv("ADMIN_WHATSAPP_NUMBER", ""),
            messenger_page_access_token=os.getenv("MESSENGER_PAGE_ACCESS_TOKEN", ""),
            instagram_access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN", ""),
            instagram_account_id=os.getenv("INSTAGRAM_ACCOUNT_ID", ""),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )

    def estado_canales(self) -> Dict[str, bool]:
        """Indica que canales de Meta tienen credenciales listas para envio real."""
        return {
            "WhatsApp": bool(
                self.whatsapp_access_token and self.whatsapp_phone_number_id
            ),
            "Facebook": bool(self.messenger_page_access_token),
            "Instagram": bool(self.instagram_access_token),
            "AdminAlertaWhatsApp": bool(
                self.whatsapp_access_token
                and self.whatsapp_phone_number_id
                and self.admin_whatsapp_number
            ),
        }

