"""
Clase base para los 5 agentes MAS-ML construidos sobre `spade.agent.Agent`.
Provee gestion de creencias, planes conceptuales, validacion de roles y
emision de mensajes FIPA-ACL monitoreados en tiempo real.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from src.comunicacion.fipa_acl import ONTOLOGIA_COMERCIAL, crear_mensaje_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.organizacion.roles import RolMASML

JID_COORDINADOR = "coordinador@localhost"
JID_VENTAS = "ventas@localhost"
JID_COTIZACIONES = "cotizaciones@localhost"
JID_PROMOCIONES = "promociones@localhost"
JID_AGENDAMIENTOS = "agendamientos@localhost"


class AgenteBaseMASML(Agent):
    """
    Base comÃºn para todos los agentes SPADE del SistemaMultiagenteComercial.
    Integra la arquitectura MAS-ML (`<<agent>>` jugando un `<<role>>` en una
    `<<organization>>` que habita un `<<environment>>`).
    """

    def __init__(
        self,
        jid: str,
        password: str,
        nombre_agente: str,
        nombre_rol: str,
        arquitectura_tipo: str,
        entorno: EntornoVirtualAtencion,
        organizacion: SistemaMultiagenteComercial,
        monitor: MonitorFIPAACL,
    ) -> None:
        super().__init__(jid, password, verify_security=False)
        self.nombre_agente = nombre_agente
        self.nombre_rol = nombre_rol
        self.arquitectura_tipo = arquitectura_tipo
        self.entorno = entorno
        self.organizacion = organizacion
        self.monitor = monitor
        self.rol: RolMASML = self.organizacion.asignar_agente_a_rol(
            nombre_agente=self.nombre_agente, nombre_rol=self.nombre_rol
        )
        self.creencias: Dict[str, Any] = {}
        self.objetivos: List[str] = []
        self.planes: Dict[str, Any] = {}
        self.evento_ciclo_completado: asyncio.Event = asyncio.Event()
        self.ultimo_resultado: Dict[str, Any] = {}

    def reiniciar_evento_ciclo(self) -> None:
        """Limpia el evento de sincronizaciÃ³n antes de iniciar un nuevo flujo."""
        self.evento_ciclo_completado = asyncio.Event()
        self.ultimo_resultado = {}

    def marcar_ciclo_completado(self, resultado: Optional[Dict[str, Any]] = None) -> None:
        """SeÃ±aliza al orquestador que el flujo de este agente culminÃ³."""
        if resultado is not None:
            self.ultimo_resultado = resultado
        self.evento_ciclo_completado.set()

    def registrar_accion(self, nombre_accion: str, detalle: str = "") -> None:
        """Registra en el monitor una acciÃ³n ejecutada por el agente."""
        self.monitor.registrar_accion_agente(
            agente=self.nombre_agente, accion=nombre_accion, detalle=detalle
        )

    @staticmethod
    def crear_plantilla_comercial() -> Template:
        """Crea la plantilla SPADE filtrando por ontologÃ­a Comercial-SMA."""
        plantilla = Template()
        plantilla.set_metadata("ontology", ONTOLOGIA_COMERCIAL)
        return plantilla

    async def enviar_fipa(
        self,
        comportamiento: CyclicBehaviour,
        receptor_jid: str,
        performativa: str,
        predicado_contenido: str,
        protocolo: str = "FIPA-Request",
        conversation_id: Optional[str] = None,
        payload_extra: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Construye, monitorea en tiempo real y envÃ­a un `spade.message.Message`
        con metadatos FIPA-ACL explÃ­citos.
        """
        msg = crear_mensaje_fipa(
            sender_jid=str(self.jid),
            receiver_jid=receptor_jid,
            performative=performativa,
            predicado_contenido=predicado_contenido,
            protocol=protocolo,
            conversation_id=conversation_id,
            payload_extra=payload_extra,
        )
        self.monitor.registrar_mensaje_fipa(msg)
        await comportamiento.send(msg)
        return msg

