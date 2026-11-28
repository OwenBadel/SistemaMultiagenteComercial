"""
Punto de entrada principal de la plataforma comercial multi-negocio.
Ejecuta FastAPI + Uvicorn con Google Gemini Flash (Tool Calling), Supabase/SQLite,
Google Calendar independiente por negocio y Panel PWA movil de alto contraste.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.comercial.configuracion import ConfiguracionComercial
from src.comercial.servidor_comercial import crear_app_comercial, crear_repositorio_comercial


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plataforma Comercial Multi-Negocio (Gemini Flash + WhatsApp + Panel PWA)"
    )
    parser.add_argument("--host", default=None, help="Host HTTP de escucha (default: HOST en .env o 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Puerto HTTP (default: PORT en .env o 8000)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _construir_parser().parse_args(argv)
    config = ConfiguracionComercial.desde_entorno()
    host = args.host or config.host
    port = args.port or config.port

    repo = crear_repositorio_comercial(config)
    app = crear_app_comercial(config=config, repositorio=repo)

    consola = Console()
    tabla = Table(show_header=False, box=None, padding=(0, 2))
    tabla.add_column("Campo", style="bold cyan")
    tabla.add_column("Valor", style="white")
    tabla.add_row("Panel PWA (Admin y Duenos)", f"http://127.0.0.1:{port}/")
    tabla.add_row("Webhook Botpress Multi-Negocio", f"POST http://127.0.0.1:{port}/api/webhook/{{negocio_id}}")
    tabla.add_row("Webhook Meta WhatsApp Cloud API", f"GET/POST http://127.0.0.1:{port}/api/whatsapp/webhook")
    tabla.add_row("Base de Datos", repo.modo_bd)
    tabla.add_row("Negocios Activos", str(len(repo.listar_negocios())))
    tabla.add_row("Motor IA", f"{config.gemini_model} ({'ACTIVO' if config.usa_gemini else 'Falta GEMINI_API_KEY'})")
    consola.print(Panel(tabla, title="Plataforma Comercial Multi-Negocio", border_style="bright_yellow"))

    avisos = config.advertencias_seguridad()
    if avisos:
        consola.print(
            Panel(
                "\n".join(f"- {a}" for a in avisos),
                title="Advertencias de Seguridad",
                border_style="bold red",
            )
        )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
