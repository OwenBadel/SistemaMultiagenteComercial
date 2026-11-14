"""
Punto de entrada principal de SistemaMultiagenteComercial.
Ejecuta el sistema sobre el servidor XMPP embebido de SPADE (`embedded_xmpp_server=True`)
con un solo comando: `python main.py` (sin requerir Prosody ni Ejabberd externos).

Autoría: Owen Badel Hooker — Ingeniero de Sistemas
Equipo Académico: Owen Badel Hooker, Maverick Díaz, Jesús Viloria
Asignatura: Inteligencia Artificial | Profesor: Juan Puello
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

import spade
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.escenarios.orquestador_escenarios import SistemaComercialOrquestador


def _silenciar_logs_internos_xmpp() -> None:
    """Evita ruido excesivo de PyJabber/Slixmpp para destacar el Monitor FIPA-ACL."""
    for nombre_logger in ("pyjabber", "slixmpp", "slixmpp_multiplatform", "aiohttp", "SPADE"):
        logging.getLogger(nombre_logger).setLevel(logging.WARNING)


def _imprimir_banner(consola: Console, modo_xmpp: bool) -> None:
    modo_str = (
        "SPADE XMPP Embebido (PyJabber localhost:5222)"
        if modo_xmpp
        else "Contenedor Local SPADE en Memoria"
    )
    consola.print(
        Panel.fit(
            "[bold cyan]SistemaMultiagenteComercial[/bold cyan] · [white]MAS-ML + FIPA-ACL + SPADE + Scikit-Learn[/white]\n"
            f"[green]Transporte:[/green] {modo_str}\n"
            "[yellow]Autoría:[/yellow] Owen Badel Hooker · Maverick Díaz · Jesús Viloria\n"
            "[dim]Asignatura: Inteligencia Artificial · Profesor: Juan Puello · Unicolombo[/dim]",
            border_style="cyan",
        )
    )


def _imprimir_menu(consola: Console) -> None:
    tabla = Table(
        title="Modo Demostración Interactiva — Seleccione un Escenario",
        show_header=True,
        header_style="bold magenta",
    )
    tabla.add_column("Opción", style="bold cyan", justify="center")
    tabla.add_column("Escenario / Flujo FIPA-ACL", style="white")
    tabla.add_column("Mecanismo IA / Protocolo", style="green")

    tabla.add_row("1", "Caso 1A: Cotización exitosa", "FIPA-Request · RandomForestRegressor -> INFORM")
    tabla.add_row("2", "Caso 1B: Cotización con parámetros incompletos", "FIPA-Request -> AGREE -> FAILURE -> REQUEST")
    tabla.add_row("3", "Caso 2A: Promoción aprobada con Regresión KNN", "FIPA-Propose · KNN -> PROPOSE -> ACCEPT-PROPOSAL")
    tabla.add_row("4", "Caso 2B: Promoción rechazada por margen mínimo", "FIPA-Propose · KNN -> REJECT-PROPOSAL -> REFUSE")
    tabla.add_row("5", "Caso 3A: Agendamiento exitoso de cita", "FIPA-Query · QUERY-IF -> AGREE -> CONFIRM")
    tabla.add_row("6", "Caso 3B: Superposición horaria dinámica", "eventoCambioDisponibilidad() -> PROPOSE alternativas")
    tabla.add_row("7", "Caso 4:  Reclamo crítico / tono hostil", "NLP Hostil -> INFORM -> CANCEL -> Alerta Humana")
    tabla.add_row("8", "Ejecutar TODOS los escenarios en secuencia (1A-4)", "Demostración integral completa")
    tabla.add_row("9", "Probar mensaje personalizado en vivo", "NLP + FIPA-ACL en tiempo real")
    tabla.add_row("0", "Salir y detener agentes SPADE", "Apagado limpio del sistema")
    consola.print(tabla)


async def _ejecutar_opcion(
    orq: SistemaComercialOrquestador, opcion: str, consola: Console
) -> bool:
    """Despacha la opción elegida en el menú. Retorna False si debe salir."""
    op = opcion.strip().upper()
    if op in {"0", "SALIR", "EXIT", "Q"}:
        return False
    if op in {"1", "1A"}:
        await orq.ejecutar_caso_1a()
    elif op in {"2", "1B"}:
        await orq.ejecutar_caso_1b()
    elif op in {"3", "2A"}:
        await orq.ejecutar_caso_2a()
    elif op in {"4", "2B"}:
        await orq.ejecutar_caso_2b()
    elif op in {"5", "3A"}:
        await orq.ejecutar_caso_3a()
    elif op in {"6", "3B"}:
        await orq.ejecutar_caso_3b()
    elif op in {"7", "4"}:
        await orq.ejecutar_caso_4()
    elif op in {"8", "ALL"}:
        await orq.ejecutar_caso_1a()
        await orq.ejecutar_caso_1b()
        await orq.ejecutar_caso_2a()
        await orq.ejecutar_caso_2b()
        await orq.ejecutar_caso_3a()
        await orq.ejecutar_caso_3b()
        await orq.ejecutar_caso_4()
    elif op == "9":
        texto = await asyncio.to_thread(
            input,
            "\nEscribe el mensaje del cliente (ej. 'Cotizar app_movil complejidad 4 con 60 horas'): ",
        )
        if texto.strip():
            await orq.ejecutar_mensaje_personalizado(texto_cliente=texto.strip())
    else:
        consola.print(f"[bold red]Opción inválida: {opcion}[/bold red]")
    return True


async def ejecutar_sistema(
    usar_xmpp_embebido: bool = True,
    demo_all: bool = False,
    caso_unico: Optional[str] = None,
) -> None:
    """Corrutina principal gestionada por SPADE."""
    consola = Console()
    _imprimir_banner(consola, modo_xmpp=usar_xmpp_embebido)

    orq = SistemaComercialOrquestador(
        usar_xmpp_embebido=usar_xmpp_embebido,
        consola=consola,
        silencioso=False,
    )
    await orq.iniciar()
    orq.imprimir_resumen_inicial()

    try:
        if demo_all:
            await _ejecutar_opcion(orq, "8", consola)
            return

        if caso_unico:
            await _ejecutar_opcion(orq, caso_unico, consola)
            return

        while True:
            _imprimir_menu(consola)
            try:
                eleccion = await asyncio.to_thread(
                    input, "\nSeleccione una opción [0-9]: "
                )
            except EOFError:
                break
            continuar = await _ejecutar_opcion(orq, eleccion, consola)
            if not continuar:
                break
    finally:
        await orq.detener()
        consola.print("[bold green]✅ Agentes SPADE detenidos correctamente.[/bold green]")


def parsear_argumentos(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SistemaMultiagenteComercial — Demostración SPADE + Scikit-Learn + FIPA-ACL"
    )
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Ejecuta automáticamente todos los escenarios (1A, 1B, 2A, 2B, 3A, 3B, 4) y finaliza.",
    )
    parser.add_argument(
        "--caso",
        type=str,
        choices=["1A", "1B", "2A", "2B", "3A", "3B", "4"],
        help="Ejecuta un escenario específico de forma directa.",
    )
    parser.add_argument(
        "--local-bus",
        action="store_true",
        help="Usa el contenedor local SPADE sin iniciar el servidor PyJabber XMPP en el puerto 5222.",
    )
    return parser.parse_args(argv)


def main() -> None:
    _silenciar_logs_internos_xmpp()
    args = parsear_argumentos(sys.argv[1:])
    usar_xmpp = not args.local_bus

    if usar_xmpp:
        spade.run(
            ejecutar_sistema(
                usar_xmpp_embebido=True,
                demo_all=args.demo_all,
                caso_unico=args.caso,
            ),
            embedded_xmpp_server=True,
        )
    else:
        asyncio.run(
            ejecutar_sistema(
                usar_xmpp_embebido=False,
                demo_all=args.demo_all,
                caso_unico=args.caso,
            )
        )


if __name__ == "__main__":
    main()
