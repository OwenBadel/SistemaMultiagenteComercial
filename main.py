"""
Punto de entrada principal de SistemaMultiagenteComercial.

Ejecuta el servidor productivo unificado:
1. Servidor XMPP embebido de SPADE (PyJabber) y ciclo de vida de los 5 agentes
   MAS-ML (AgenteCoordinador, AgenteVentas, AgenteCotizaciones, AgentePromociones
   y AgenteAgendamientos).
2. Entorno compartido EntornoVirtualAtencion integrado con Botpress Studio
   (puente hacia WhatsApp, Webchat, Facebook Messenger e Instagram).
3. Servidor HTTP FastAPI + Uvicorn con el endpoint para Botpress (/api/botpress/mensaje)
   y el panel web Human-in-the-Loop y Monitor FIPA-ACL en tiempo real (/).
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
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.integracion_botpress.servidor_botpress import crear_app_botpress


def _construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos del servidor."""
    parser = argparse.ArgumentParser(
        description=(
            "Servidor productivo SistemaMultiagenteComercial "
            "(SPADE + Scikit-Learn + Botpress)."
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
    config_botpress: ConfiguracionBotpress,
    servicio: ServicioMultiagenteComercial,
    usar_xmpp: bool,
) -> None:
    """Muestra en consola los endpoints activos para Botpress, BD y Monitor FIPA-ACL."""
    host_visible = "127.0.0.1" if config_botpress.host == "0.0.0.0" else config_botpress.host
    cfg_neg = servicio.entorno.config_negocio

    tabla = Table(show_header=True, header_style="bold cyan")
    tabla.add_column("Componente", style="bold")
    tabla.add_column("Estado / Valor")

    tabla.add_row(
        "Transporte SPADE",
        "XMPP Embebido (PyJabber localhost:5222)" if usar_xmpp else "Bus local asyncio",
    )
    tabla.add_row("Base de Datos", servicio.repositorio.modo_bd)
    tabla.add_row("Agenda", servicio.entorno.agendaVirtual.modo)
    tabla.add_row(
        "Negocio Activo (BD)",
        f"{cfg_neg.get('nombre_negocio')} [{servicio.repositorio.negocio_id}]",
    )
    tabla.add_row("Panel Web / Monitor FIPA-ACL", f"http://{host_visible}:{config_botpress.port}/")
    tabla.add_row(
        "Endpoint Botpress (POST)",
        f"http://{host_visible}:{config_botpress.port}/api/botpress/mensaje",
    )
    tabla.add_row(
        "Seguridad API Key (.env)",
        "ACTIVA" if config_botpress.botpress_api_key else "ABIERTA (Ideal pruebas / ngrok)",
    )

    consola.print(
        Panel(
            tabla,
            title="SistemaMultiagenteComercial - Servicio Activo",
            subtitle="SPADE + Scikit-Learn + Botpress + Supabase + Google Calendar",
            border_style="cyan",
        )
    )


async def _ejecutar_servidor(
    config_botpress: ConfiguracionBotpress,
    usar_xmpp_embebido: bool,
) -> None:
    """Inicializa los 5 agentes SPADE y sirve FastAPI + Uvicorn hasta recibir cierre."""
    consola = Console()
    servicio = ServicioMultiagenteComercial(
        config_botpress=config_botpress,
        usar_xmpp_embebido=usar_xmpp_embebido,
        consola=consola,
        silencioso=False,
    )

    await servicio.iniciar()
    servicio.imprimir_resumen_inicial()
    _imprimir_estado_servidor(consola, config_botpress, servicio, usar_xmpp_embebido)

    app = crear_app_botpress(servicio)
    uvicorn_config = uvicorn.Config(
        app=app,
        host=config_botpress.host,
        port=config_botpress.port,
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

    config_botpress = ConfiguracionBotpress.desde_entorno(RAIZ_PROYECTO / ".env")
    if args.host:
        config_botpress.host = args.host
    if args.port:
        config_botpress.port = args.port

    usar_xmpp = not args.local_bus
    try:
        if usar_xmpp:
            spade.run(
                _ejecutar_servidor(config_botpress, usar_xmpp_embebido=True),
                embedded_xmpp_server=True,
            )
        else:
            asyncio.run(_ejecutar_servidor(config_botpress, usar_xmpp_embebido=False))
    except KeyboardInterrupt:
        print("\nServidor detenido por el operador.")


if __name__ == "__main__":
    main()
