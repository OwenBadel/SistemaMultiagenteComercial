"""
Implementación del AgenteAgendamientos (<<play>> rol GestorAgenda).
Arquitectura: Basado en metas.
Previene superposiciones horarias, consulta disponibilidad real en agendaVirtual,
sugiere bloques alternativos y registra citas confirmadas vía QUERY-IF, AGREE y CONFIRM.

Autoría: Owen Badel Hooker
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from src.agentes.base_agente import JID_COORDINADOR, AgenteBaseMASML
from src.comunicacion.fipa_acl import extraer_payload_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


@dataclass
class PlanVerificacionAgenda:
    """Plan MAS-ML para consultar disponibilidad y detectar cruces horarios."""

    nombre: str = "PlanVerificacionAgenda"
    objetivo: str = "Validar bloque horario sin conflictos en agendaVirtual"


@dataclass
class PlanRegistroCita:
    """Plan MAS-ML para reservar definitivamente la cita y notificar confirmación."""

    nombre: str = "PlanRegistroCita"
    objetivo: str = "Escribir la cita en agendaVirtual y confirmar al Coordinador"


class AgenteAgendamientos(AgenteBaseMASML):
    """
    <<agent>> AgenteAgendamientos:
    Agente basado en metas encargado de verificar y registrar citas en el
    Calendario del EntornoVirtualAtencion.
    """

    def __init__(
        self,
        entorno: EntornoVirtualAtencion,
        organizacion: SistemaMultiagenteComercial,
        monitor: MonitorFIPAACL,
        jid: str = "agendamientos@localhost",
        password: str = "sma_2026",
    ) -> None:
        super().__init__(
            jid=jid,
            password=password,
            nombre_agente="AgenteAgendamientos",
            nombre_rol="GestorAgenda",
            arquitectura_tipo="Basado en metas",
            entorno=entorno,
            organizacion=organizacion,
            monitor=monitor,
        )
        self.creencias: Dict[str, Any] = {
            "fechaHoraSolicitada": None,
            "estadoCalendario": {},
            "confirmacionCotizacion": False,
        }
        self.objetivos = [
            "Garantizar reserva efectiva sin cruces de horario",
        ]
        self.planes = {
            "PlanVerificacionAgenda": PlanVerificacionAgenda(),
            "PlanRegistroCita": PlanRegistroCita(),
        }

    class ComportamientoGestorAgenda(CyclicBehaviour):
        """Escucha QUERY-IF y REQUEST provenientes del AgenteCoordinador."""

        async def run(self) -> None:
            msg: Message | None = await self.receive(timeout=0.2)
            if msg is None:
                return
            await self.agent.procesar_mensaje_agenda(self, msg)

    async def setup(self) -> None:
        self.add_behaviour(
            self.ComportamientoGestorAgenda(), self.crear_plantilla_comercial()
        )

    # ------------------------------------------------------------------
    # ACCIONES DEL DIAGRAMA DE CLASES MAS-ML
    # ------------------------------------------------------------------
    def consultarCalendario(self, fecha_hora: str) -> Dict[str, Any]:
        """Consulta el estado del calendario en EntornoVirtualAtencion."""
        self.creencias["fechaHoraSolicitada"] = fecha_hora
        estado = self.entorno.consultarCalendario(fecha_hora)
        self.creencias["estadoCalendario"] = estado
        self.registrar_accion(
            "consultarCalendario()", f"{fecha_hora} -> {estado['estado']}"
        )
        return estado

    def sugerirHorariosAlternativos(self, fecha_hora_ocupada: str) -> List[str]:
        """Obtiene hasta dos bloques libres alternativos cuando hay cruce horario."""
        alternativas = self.entorno.agendaVirtual.obtener_alternativas(
            excluir_fecha=fecha_hora_ocupada, limite=2
        )
        self.registrar_accion(
            "sugerirHorariosAlternativos()", f"alternativas={alternativas}"
        )
        return alternativas

    def administrarCitaYNotificar(self, fecha_hora: str, nombre_cliente: str) -> Dict[str, Any]:
        """Registra la cita en agendaVirtual y notifica el éxito de reserva."""
        self.registrar_accion(
            "administrarCitaYNotificar()", f"{fecha_hora} | cliente={nombre_cliente}"
        )
        return self.entorno.registrarCitaEnCalendario(
            fecha_hora=fecha_hora, cliente=nombre_cliente, estado="OCUPADO"
        )

    async def procesar_mensaje_agenda(
        self, comportamiento: CyclicBehaviour, msg: Message
    ) -> None:
        perf = (msg.get_metadata("performative") or "").upper()
        conv_id = msg.get_metadata("conversation_id") or msg.thread or "conv-default"
        payload = extraer_payload_fipa(msg)
        fecha_hora = str(payload.get("fecha_hora", "2026-11-18 10:00"))

        if perf == "QUERY-IF":
            self.creencias["confirmacionCotizacion"] = bool(
                payload.get("cotizacion_aprobada", False)
            )
            if not self.organizacion.validar_prerequisito_agendamiento(
                self.creencias["confirmacionCotizacion"]
            ):
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: cotizacion_no_aprobada)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                )
                return

            estado = self.consultarCalendario(fecha_hora)
            if estado["disponible"]:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "INFORM",
                    "(disponible_bloque fecha_hora true)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**payload, "disponible": True},
                )
            else:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "INFORM",
                    "(disponible_bloque fecha_hora false)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**payload, "disponible": False},
                )
                alternativas = self.sugerirHorariosAlternativos(fecha_hora)
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "PROPOSE",
                    "(horarios_alternativos bloque_A bloque_B)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={
                        **payload,
                        "alternativas": alternativas,
                    },
                )

        elif perf == "REQUEST":
            nombre_cliente = str(payload.get("nombre_cliente", "Cliente Empresarial"))
            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "AGREE",
                "(registrar_cita cliente fecha_hora)",
                protocolo="FIPA-Query",
                conversation_id=conv_id,
                payload_extra=payload,
            )
            self.administrarCitaYNotificar(fecha_hora=fecha_hora, nombre_cliente=nombre_cliente)
            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "CONFIRM",
                "(cita_registrada fecha_hora)",
                protocolo="FIPA-Query",
                conversation_id=conv_id,
                payload_extra={
                    **payload,
                    "fecha_hora": fecha_hora,
                    "estado": "OCUPADO",
                },
            )
