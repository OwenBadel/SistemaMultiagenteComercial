"""
Configuracion del puente API entre Botpress y SistemaMultiagenteComercial.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class ConfiguracionBotpress:
    """Parametros del servidor HTTP conectado con Botpress Studio."""

    botpress_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def desde_entorno(cls, ruta_env: Path | None = None) -> "ConfiguracionBotpress":
        """Carga variables opcionales desde `.env` y el entorno del sistema."""
        if ruta_env is not None and ruta_env.exists():
            load_dotenv(dotenv_path=ruta_env, override=False)
        else:
            load_dotenv(override=False)

        return cls(
            botpress_api_key=os.getenv("BOTPRESS_API_KEY", ""),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )
