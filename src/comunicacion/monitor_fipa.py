"""
Logger / Monitor FIPA-ACL en tiempo real con Rich.
Imprime cada mensaje entre agentes tanto con la etiqueta de 3 lineas (+ estereotipo)
como en estructura FIPA-ACL (JSON con :sender, :receiver, :content, :ontology, :protocol).

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from spade.message import Message

from src.comunicacion.fipa_acl import (
    formatear_etiqueta_diapositiva,
    serializar_estructura_fipa_json,
)

_COLORES_PERFORMATIVA = {
    "REQUEST": "bold cyan",
    "AGREE": "bold green",
    "INFORM": "bold bright_blue",
    "FAILURE": "bold red",
    "PROPOSE": "bold magenta",
    "ACCEPT-PROPOSAL": "bold bright_green",
    "REJECT-PROPOSAL": "bold yellow",
    "REFUSE": "bold red",
    "QUERY-IF": "bold bright_cyan",
    "CONFIRM": "bold green",
    "CANCEL": "bold bright_red",
}


class MonitorFIPAACL:
    """
    Monitor y registrador en tiempo real de mensajes FIPA-ACL, eventos de entorno
    y acciones de los agentes del SistemaMultiagenteComercial.
    """

    def __init__(self, consola: Optional[Console] = None, silencioso: bool = False) -> None:
        self.consola: Console = consola or Console()
        self.silencioso: bool = silencioso
        self.historial_mensajes: List[Dict[str, Any]] = []
        self.historial_eventos_entorno: List[Dict[str, Any]] = []
        self.historial_acciones: List[Dict[str, Any]] = []

    def limpiar(self) -> None:
        """Reinicia las trazas acumuladas."""
        self.historial_mensajes.clear()
        self.historial_eventos_entorno.clear()
        self.historial_acciones.clear()

    def registrar_mensaje_fipa(self, msg: Message) -> Dict[str, Any]:
        """
        Registra un `spade.message.Message` y lo imprime en tiempo real con
        ambos formatos exigidos (diapositiva de clase + estructura JSON FIPA).
        """
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        etiqueta_slide = formatear_etiqueta_diapositiva(msg)
        estructura_fipa = serializar_estructura_fipa_json(msg, timestamp=timestamp)
        perf = estructura_fipa[":performative"]

        registro = {
            "etiqueta_diapositiva": etiqueta_slide,
            "fipa_json": estructura_fipa,
        }
        self.historial_mensajes.append(registro)

        if not self.silencioso:
            estilo = _COLORES_PERFORMATIVA.get(perf, "bold white")
            cuerpo = Text()
            cuerpo.append(etiqueta_slide + "\n", style=estilo)
            cuerpo.append("-" * 58 + "\n", style="dim")
            cuerpo.append(
                json.dumps(estructura_fipa, ensure_ascii=False, indent=2),
                style="white",
            )
            self.consola.print(
                Panel(
                    cuerpo,
                    title=f"[bold]FIPA-ACL Monitor | {estructura_fipa[':protocol']}[/bold]",
                    border_style=estilo.replace("bold ", ""),
                    expand=False,
                )
            )
        return registro

    def registrar_evento_entorno(self, origen: str, detalle: str) -> None:
        """Registra e imprime un evento disparado desde o hacia EntornoVirtualAtencion."""
        registro = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "origen": origen,
            "detalle": detalle,
        }
        self.historial_eventos_entorno.append(registro)
        if not self.silencioso:
            self.consola.print(f"[bold yellow][ENTORNO][/bold yellow] [{origen}]: {detalle}")

    def registrar_accion_agente(self, agente: str, accion: str, detalle: str = "") -> None:
        """Registra una accion interna ejecutada por un agente MAS-ML."""
        registro = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "agente": agente,
            "accion": accion,
            "detalle": detalle,
        }
        self.historial_acciones.append(registro)
        if not self.silencioso:
            info = f" ({detalle})" if detalle else ""
            self.consola.print(f"[bold green][ACCION][/bold green] [{agente}]: {accion}{info}")
