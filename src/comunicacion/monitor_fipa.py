"""
Logger / Monitor FIPA-ACL en tiempo real con Rich.
Imprime cada mensaje entre agentes tanto con la etiqueta de 3 lineas (+ estereotipo)
como en estructura FIPA-ACL (JSON con :sender, :receiver, :content, :ontology, :protocol).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from spade.message import Message

from src.comunicacion.fipa_acl import (
    ONTOLOGIA_COMERCIAL,
    PERFORMATIVAS_SOPORTADAS,
    PROTOCOLOS_VALIDOS,
    formatear_nombre_agente,
    serializar_estructura_fipa_json,
)

_ACCIONES_RESUMIBLES = frozenset(
    {
        "aceptar_cotizacion",
        "alerta_caso_critico",
        "aplicar_descuento",
        "aprobar_descuento",
        "calcular_costo",
        "calcular_descuento",
        "cancelar_accion",
        "cita_registrada",
        "cotizacion_calculada",
        "disponible_bloque",
        "entregar_cotizacion",
        "error_agendamiento",
        "escalar_caso",
        "evaluar_incentivo",
        "horarios_alternativos",
        "motivo",
        "ofrecer_alternativas",
        "promocion_autorizada",
        "registrar_cita",
        "rechazar_descuento",
        "reserva_confirmada",
        "solicitar_parametros_faltantes",
        "transferir_promocion",
    }
)
_AGENTES_CONOCIDOS = frozenset(
    {
        "AgenteAgendamientos",
        "AgenteCoordinador",
        "AgenteCotizaciones",
        "AgentePromociones",
        "AgenteVentas",
    }
)
_PATRON_ACCION = re.compile(r"^\s*\(\s*([a-z][a-z0-9_-]*)\b", re.IGNORECASE)

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
        self._clave_conversation_id = secrets.token_bytes(32)

    @staticmethod
    def _nombre_agente_seguro(jid: Any) -> str:
        nombre = formatear_nombre_agente(str(jid or ""))
        return nombre if nombre in _AGENTES_CONOCIDOS else "AgenteNoIdentificado"

    def _conversation_id_opaco(self, conversation_id: Any) -> str:
        valor = str(conversation_id or "sin-conversation-id")
        digest = hmac.new(
            self._clave_conversation_id,
            valor.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:12]
        return f"conv-{digest}"

    @staticmethod
    def _resumir_contenido_seguro(contenido: Any) -> str:
        coincidencia = _PATRON_ACCION.match(str(contenido or ""))
        accion = coincidencia.group(1).lower() if coincidencia else ""
        if accion not in _ACCIONES_RESUMIBLES:
            return "accion=mensaje"
        return f"accion={accion}"

    def registrar_mensaje_fipa(self, msg: Message) -> Dict[str, Any]:
        """
        Registra un mensaje cuyo envío SPADE ya terminó y muestra un resumen seguro.
        """
        timestamp = datetime.now(ZoneInfo("America/Bogota")).isoformat(
            timespec="milliseconds"
        )
        estructura_fipa = serializar_estructura_fipa_json(msg, timestamp=timestamp)
        emisor = self._nombre_agente_seguro(msg.sender)
        receptor = self._nombre_agente_seguro(msg.to)
        perf_original = str(estructura_fipa[":performative"]).upper()
        perf = (
            perf_original
            if perf_original in PERFORMATIVAS_SOPORTADAS
            else "UNKNOWN"
        )
        protocolo_original = str(estructura_fipa[":protocol"])
        protocolo = (
            protocolo_original
            if protocolo_original in PROTOCOLOS_VALIDOS
            else "FIPA"
        )
        ontologia = (
            ONTOLOGIA_COMERCIAL
            if estructura_fipa[":ontology"] == ONTOLOGIA_COMERCIAL
            else "ontology"
        )
        conversation_id = self._conversation_id_opaco(
            estructura_fipa[":conversation-id"]
        )
        resumen = self._resumir_contenido_seguro(msg.body)

        estructura_fipa.update(
            {
                ":performative": perf,
                ":sender": emisor,
                ":receiver": receptor,
                ":content": resumen,
                ":ontology": ontologia,
                ":protocol": protocolo,
                ":conversation-id": conversation_id,
            }
        )
        etiqueta_slide = (
            f"[{emisor} -> {receptor}]\n{perf}\n({resumen})\n"
            f"<<ACL:{perf}>>"
        )

        registro = {
            "etiqueta_diapositiva": etiqueta_slide,
            "fipa_json": estructura_fipa,
            "resultado_envio": "send_ok_local",
        }
        self.historial_mensajes.append(registro)

        if not self.silencioso:
            estilo = _COLORES_PERFORMATIVA.get(perf, "bold white")
            datos_panel = {
                "performative": perf,
                "sender": emisor,
                "receiver": receptor,
                "content": resumen,
                "ontology": ontologia,
                "protocol": protocolo,
                "conversation-id": conversation_id,
                "timestamp": timestamp,
            }
            self.consola.print(
                Panel(
                    Group(
                        Text(f"{emisor} -> {receptor}", style="bold"),
                        Text(perf, style=estilo),
                        Text(f"<ACL:{perf}>"),
                        Syntax(
                            json.dumps(datos_panel, ensure_ascii=False, indent=2),
                            "json",
                            word_wrap=True,
                        ),
                        Text(
                            "Resultado: send_ok_local (recepción no confirmada)",
                            style="dim",
                        ),
                    ),
                    title=Text(f"[FIPA-ACL Monitor | {protocolo}]"),
                    border_style=estilo,
                )
            )
        return registro

    def registrar_evento_entorno(self, origen: str, detalle: str) -> None:
        """Registra e imprime un evento disparado desde o hacia EntornoVirtualAtencion."""
        registro = {
            "timestamp": datetime.now(ZoneInfo("America/Bogota")).isoformat(
                timespec="milliseconds"
            ),
            "origen": origen,
            "detalle": detalle,
        }
        self.historial_eventos_entorno.append(registro)
        if not self.silencioso:
            self.consola.print(f"[bold yellow][ENTORNO][/bold yellow] [{origen}]: {detalle}")

    def registrar_accion_agente(self, agente: str, accion: str, detalle: str = "") -> None:
        """Registra una accion interna ejecutada por un agente MAS-ML."""
        registro = {
            "timestamp": datetime.now(ZoneInfo("America/Bogota")).isoformat(
                timespec="milliseconds"
            ),
            "agente": agente,
            "accion": accion,
            "detalle": detalle,
        }
        self.historial_acciones.append(registro)
        if not self.silencioso:
            info = f" ({detalle})" if detalle else ""
            self.consola.print(f"[bold green][ACCION][/bold green] [{agente}]: {accion}{info}")
