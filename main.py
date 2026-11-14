"""
Punto de entrada principal de SistemaMultiagenteComercial.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
Institucion: Fundacion Internacional Colombo (Unicolombo)

Ejecuta el servidor productivo unificado:
1. Servidor XMPP embebido de SPADE (PyJabber) y ciclo de vida de los 5 agentes
   MAS-ML (AgenteCoordinador, AgenteVentas, AgenteCotizaciones, AgentePromociones
   y AgenteAgendamientos).
2. Entorno compartido EntornoVirtualAtencion conectado a Meta Graph API
   (WhatsApp Cloud API, Instagram Direct y Facebook Messenger).
3. Servidor HTTP FastAPI + Uvicorn con el webhook oficial (/webhook) y el panel
   web Human-in-the-Loop y Monitor FIPA-ACL en tiempo real (/).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import spade
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

RAIZ_PROYECTO = Path(__file__).resolve().parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial
from src.integracion_meta.configuracion import ConfiguracionMeta
from src.integracion_meta.webhook_server import crear_app_meta


def _construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos del servidor."""
    parser = argparse.ArgumentParser(
        description=(
            "Servidor productivo SistemaMultiagenteComercial "
            "(SPADE + Scikit-Learn + Meta Cloud API)."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Direccion IP de escucha HTTP (por defecto usa HOST en .env o 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Puerto HTTP de escucha (por defecto usa PORT en .env o 8000).",
    )
    parser.add_argument(
        "--local-bus",
        action="store_true",
        help="Desactiva el servidor XMPP embebido y usa el bus local FIPA-ACL de respaldo.",
    )
    return parser


def _imprimir_estado_servidor(
    consola: Console,
    config_meta: ConfiguracionMeta,
    usar_xmpp: bool,
) -> None:
    """Muestra en consola el estado de canales Meta, endpoints y transporte FIPA-ACL."""
    host_visible = "127.0.0.1" if config_meta.host == "0.0.0.0" else config_meta.host

    tabla = Table(show_header=True, header_style="bold cyan")
    tabla.add_column("Componente", style="bold")
    tabla.add_column("Estado / Valor")

    tabla.add_row(
        "Transporte SPADE",
        "XMPP Embebido (PyJabber localhost:5222)" if usar_xmpp else "Bus local asyncio",
    )
    tabla.add_row("Panel Web / Monitor FIPA-ACL", f"http://{host_visible}:{config_meta.port}/")
    tabla.add_row("Endpoint Webhook Meta", f"http://{host_visible}:{config_meta.port}/webhook")
    tabla.add_row("META_VERIFY_TOKEN", config_meta.meta_verify_token)

    estados = config_meta.estado_canales()
    for canal, configurado in estados.items():
        etiqueta = "CONFIGURADO (Envio real Graph API)" if configurado else "SIN CREDENCIALES (Modo escucha/registro)"
        tabla.add_row(f"Canal Meta: {canal}", etiqueta)

    consola.print(
        Panel(
            tabla,
            title="SistemaMultiagenteComercial - Servicio Meta Activo",
            subtitle="Autores: Owen Badel, Maverick Diaz, Jesus Viloria",
            border_style="cyan",
        )
    )


async def _ejecutar_servidor(
    config_meta: ConfiguracionMeta,
    usar_xmpp_embebido: bool,
) -> None:
    """Inicializa los 5 agentes SPADE y sirve FastAPI + Uvicorn hasta recibir cierre."""
    consola = Console()
    servicio = ServicioMultiagenteComercial(
        ruta_data=RAIZ_PROYECTO / "data",
        config_meta=config_meta,
        usar_xmpp_embebido=usar_xmpp_embebido,
        consola=consola,
        silencioso=False,
    )

    await servicio.iniciar()
    servicio.imprimir_resumen_inicial()
    _imprimir_estado_servidor(consola, config_meta, usar_xmpp_embebido)

    app = crear_app_meta(servicio)
    uvicorn_config = uvicorn.Config(
        app=app,
        host=config_meta.host,
        port=config_meta.port,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)

    try:
        await server.serve()
    finally:
        await servicio.detener()


def main() -> None:
    """Punto de entrada principal sin simulaciones ni menus por consola."""
    parser = _construir_parser()
    args = parser.parse_args()

    config_meta = ConfiguracionMeta.desde_entorno(RAIZ_PROYECTO / ".env")
    if args.host:
        config_meta.host = args.host
    if args.port:
        config_meta.port = args.port

    usar_xmpp = not args.local_bus
    try:
        if usar_xmpp:
            spade.run(
                _ejecutar_servidor(config_meta, usar_xmpp_embebido=True),
                embedded_xmpp_server=True,
            )
        else:
            asyncio.run(_ejecutar_servidor(config_meta, usar_xmpp_embebido=False))
    except KeyboardInterrupt:
        print("\nServidor detenido por el operador.")


if __name__ == "__main__":
    main()
