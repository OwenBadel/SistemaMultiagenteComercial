"""
Implementacion del AgenteAgendamientos (<<play>> rol GestorAgenda).
Arquitectura: Basado en metas.
Previene superposiciones horarias, consulta disponibilidad real en agendaVirtual,
sugiere bloques alternativos y registra citas confirmadas via QUERY-IF, AGREE y CONFIRM.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from src.agentes.base_agente import JID_COORDINADOR, AgenteBaseMASML
from src.comunicacion.fipa_acl import extraer_payload_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.persistencia.google_calendar import ErrorGoogleCalendar
from src.persistencia.repositorio import ConflictoAgendamientoError
from src.reglas_comerciales import validar_datos_cliente_reserva


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
    def consultarCalendario(
        self, fecha_hora: str, alternativas_excluidas: List[str] | None = None
    ) -> Dict[str, Any]:
        """Consulta el estado del calendario en EntornoVirtualAtencion."""
        self.creencias["fechaHoraSolicitada"] = fecha_hora
        estado = self.entorno.consultarCalendario(
            fecha_hora, alternativas_excluidas
        )
        self.creencias["estadoCalendario"] = estado
        self.registrar_accion(
            "consultarCalendario()", f"estado={estado['estado']}"
        )
        return estado

    def sugerirHorariosAlternativos(
        self, fecha_hora_ocupada: str, alternativas_excluidas: List[str] | None = None
    ) -> List[str]:
        """Obtiene hasta dos bloques libres alternativos cuando hay cruce horario."""
        alternativas = self.entorno.agendaVirtual.obtener_alternativas(
            excluir_fecha=fecha_hora_ocupada,
            limite=2,
            excluir_horarios=alternativas_excluidas,
        )
        self.registrar_accion(
            "sugerirHorariosAlternativos()",
            f"alternativas_disponibles={len(alternativas)}",
        )
        return alternativas

    def administrarCitaYNotificar(
        self, datos: Dict[str, Any], datos_cliente: Dict[str, str]
    ) -> Dict[str, Any]:
        """Persiste la cita antes de informar cualquier confirmación."""
        self.registrar_accion(
            "administrarCitaYNotificar()",
            "persistencia_solicitada",
        )
        return self.entorno.registrarCitaEnCalendario(
            fecha_hora=str(datos["fecha_hora"]),
            cliente=str(datos["nombre_cliente"]),
            estado="OCUPADO",
            id_cliente=str(datos["id_cliente"]),
            canal=str(datos["canal"]),
            servicio=str(datos["servicio"]) if datos.get("servicio") else None,
            metadata={
                "conversation_id": str(
                    datos.get("contexto_conversacion_id")
                    or datos.get("conversation_id", "")
                ),
                "fipa_conversation_id": str(datos.get("conversation_id", "")),
                "contexto_conversacion_id": str(
                    datos.get("contexto_conversacion_id") or ""
                ),
                "datos_cliente": datos_cliente,
            },
        )

    def _obtener_datos_cliente_reserva(
        self, payload: Dict[str, Any]
    ) -> Dict[str, str]:
        id_cliente = str(payload.get("id_cliente", ""))
        contexto_id = payload.get("contexto_conversacion_id")
        estado = self.entorno.perfilesClientes.obtener_estado_conversacion(
            id_cliente, str(contexto_id) if contexto_id else None
        )
        estado_reserva = estado.get("estado_reserva", {})
        if not isinstance(estado_reserva, dict):
            raise ValueError("Faltan los datos completos del cliente para la reserva.")
        return validar_datos_cliente_reserva(
            estado_reserva.get("nombre_cliente"),
            estado_reserva.get("cedula"),
            estado_reserva.get("telefono"),
        )

    def _detalle_error_persistencia(self, error: Exception) -> str:
        if not isinstance(error, httpx.HTTPStatusError):
            return f"error={type(error).__name__}"

        respuesta = error.response
        try:
            cuerpo = respuesta.json()
        except ValueError:
            cuerpo = {}
        if not isinstance(cuerpo, dict):
            cuerpo = {}

        partes = [
            f"HTTP {respuesta.status_code}",
            f"{respuesta.request.method} {respuesta.request.url.path}",
        ]
        codigo = cuerpo.get("code")
        mensaje = cuerpo.get("message")
        if isinstance(codigo, str) and codigo:
            partes.append(f"codigo={codigo}")
        if isinstance(mensaje, str) and mensaje:
            partes.append(f"mensaje={' '.join(mensaje.split())[:240]}")
        return " | ".join(partes)

    async def procesar_mensaje_agenda(
        self, comportamiento: CyclicBehaviour, msg: Message
    ) -> None:
        perf = (msg.get_metadata("performative") or "").upper()
        conv_id = msg.get_metadata("conversation_id") or msg.thread or "conv-default"
        payload = extraer_payload_fipa(msg)
        fecha_hora = str(payload.get("fecha_hora", "")).strip()

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

            try:
                datos_cliente = self._obtener_datos_cliente_reserva(payload)
            except ValueError as exc:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: datos_cliente_incompletos)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**payload, "motivo_error": str(exc)},
                )
                return

            datos_cita = {
                **payload,
                "fecha_hora": fecha_hora,
                "nombre_cliente": datos_cliente["nombre_cliente"],
                "canal": str(payload.get("canal", "WhatsApp")),
                "conversation_id": conv_id,
            }
            try:
                datos_validados = self.entorno.validarDatosAgendamiento(
                    {
                        **datos_cita,
                        "metadata": {"datos_cliente": datos_cliente},
                    }
                )
                datos_cita.update(
                    {
                        clave: valor
                        for clave, valor in datos_validados.items()
                        if clave != "metadata"
                    }
                )
                estado = self.consultarCalendario(
                    datos_cita["fecha_hora"],
                    payload.get("alternativas_rechazadas", []),
                )
            except ValueError as exc:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: datos_cita_invalidos)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**payload, "motivo_error": str(exc)},
                )
                return
            except ErrorGoogleCalendar as exc:
                try:
                    datos_pendientes = {
                        **datos_cita,
                        "metadata": {
                            "datos_cliente": datos_cliente,
                            "contexto_conversacion_id": payload.get(
                                "contexto_conversacion_id"
                            ),
                        },
                    }
                    pendiente = self.entorno.registrarAgendamientoPendiente(
                        datos_pendientes, str(exc)
                    )
                except ConflictoAgendamientoError:
                    await self._rechazar_horario_ocupado(
                        comportamiento, conv_id, datos_cita
                    )
                    return
                except (httpx.HTTPError, sqlite3.Error, RuntimeError, ValueError) as error:
                    self.registrar_accion(
                        "falloPersistenciaAgendamiento()",
                        self._detalle_error_persistencia(error),
                    )
                    await self._enviar_falla_persistencia(
                        comportamiento, conv_id, datos_cita
                    )
                    return
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "FAILURE",
                    "(motivo: google_calendar_disponibilidad_no_verificada)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={
                        **datos_cita,
                        "agendamiento_id": pendiente.get("id"),
                        "estado": "pendiente",
                        "google_calendar_estado": "error",
                    },
                )
                return
            except (httpx.HTTPError, sqlite3.Error, RuntimeError) as exc:
                self.registrar_accion(
                    "falloConsultaAgendamiento()",
                    self._detalle_error_persistencia(exc),
                )
                await self._enviar_falla_persistencia(
                    comportamiento, conv_id, payload
                )
                return

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
                alternativas = estado.get("alternativas", [])
                self.registrar_accion(
                    "sugerirHorariosAlternativos()",
                    f"alternativas={alternativas}",
                )
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
            cotizacion_aprobada = bool(
                payload.get(
                    "cotizacion_aprobada",
                    self.creencias["confirmacionCotizacion"],
                )
            )
            if not self.organizacion.validar_prerequisito_agendamiento(
                cotizacion_aprobada
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

            try:
                datos_cliente = self._obtener_datos_cliente_reserva(payload)
            except ValueError as exc:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: datos_cliente_incompletos)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**payload, "motivo_error": str(exc)},
                )
                return

            datos_cita = {
                **payload,
                "fecha_hora": fecha_hora,
                "nombre_cliente": datos_cliente["nombre_cliente"],
                "canal": str(payload.get("canal", "WhatsApp")),
                "conversation_id": conv_id,
            }
            try:
                datos_validados = self.entorno.validarDatosAgendamiento(
                    {
                        **datos_cita,
                        "metadata": {"datos_cliente": datos_cliente},
                    }
                )
                datos_cita.update(
                    {
                        clave: valor
                        for clave, valor in datos_validados.items()
                        if clave != "metadata"
                    }
                )
                estado = self.consultarCalendario(
                    datos_cita["fecha_hora"],
                    payload.get("alternativas_rechazadas", []),
                )
            except ValueError as exc:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: datos_cita_invalidos)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**datos_cita, "motivo_error": str(exc)},
                )
                return
            except ErrorGoogleCalendar as exc:
                try:
                    datos_pendientes = {
                        **datos_cita,
                        "metadata": {
                            "datos_cliente": datos_cliente,
                            "contexto_conversacion_id": payload.get(
                                "contexto_conversacion_id"
                            ),
                        },
                    }
                    pendiente = self.entorno.registrarAgendamientoPendiente(
                        datos_pendientes, str(exc)
                    )
                except ConflictoAgendamientoError:
                    await self._rechazar_horario_ocupado(
                        comportamiento, conv_id, datos_cita
                    )
                    return
                except (httpx.HTTPError, sqlite3.Error, RuntimeError, ValueError) as error:
                    self.registrar_accion(
                        "falloPersistenciaAgendamiento()",
                        self._detalle_error_persistencia(error),
                    )
                    await self._enviar_falla_persistencia(
                        comportamiento, conv_id, datos_cita
                    )
                    return
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "FAILURE",
                    "(motivo: google_calendar_disponibilidad_no_verificada)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={
                        **datos_cita,
                        "agendamiento_id": pendiente.get("id"),
                        "estado": "pendiente",
                        "google_calendar_estado": "error",
                    },
                )
                return
            except (httpx.HTTPError, sqlite3.Error, RuntimeError) as exc:
                self.registrar_accion(
                    "falloConsultaAgendamiento()",
                    self._detalle_error_persistencia(exc),
                )
                await self._enviar_falla_persistencia(
                    comportamiento, conv_id, datos_cita
                )
                return

            if not estado["disponible"]:
                await self._rechazar_horario_ocupado(
                    comportamiento, conv_id, datos_cita
                )
                return

            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "AGREE",
                "(registrar_cita cliente fecha_hora)",
                protocolo="FIPA-Query",
                conversation_id=conv_id,
                payload_extra=datos_cita,
            )
            try:
                registro = self.administrarCitaYNotificar(
                    datos_cita, datos_cliente
                )
            except ConflictoAgendamientoError:
                alternativas = self.sugerirHorariosAlternativos(
                    str(datos_cita["fecha_hora"]),
                    payload.get("alternativas_rechazadas", []),
                )
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "FAILURE",
                    "(motivo: horario_ocupado)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={**datos_cita, "alternativas": alternativas},
                )
                return
            except (httpx.HTTPError, sqlite3.Error, RuntimeError, ValueError) as exc:
                self.registrar_accion(
                    "falloPersistenciaAgendamiento()",
                    self._detalle_error_persistencia(exc),
                )
                await self._enviar_falla_persistencia(
                    comportamiento, conv_id, datos_cita
                )
                return

            if (
                registro.get("status") != "ok"
                or registro.get("estado") != "confirmada"
            ):
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "FAILURE",
                    "(motivo: disponibilidad_google_calendar_no_verificada)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={
                        **datos_cita,
                        "agendamiento_id": registro.get("id"),
                        "estado": registro.get("estado", "pendiente"),
                        "google_calendar_estado": registro.get(
                            "google_calendar_estado", "error"
                        ),
                    },
                )
                return

            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "CONFIRM",
                "(cita_registrada fecha_hora)",
                protocolo="FIPA-Query",
                conversation_id=conv_id,
                payload_extra={
                    **datos_cita,
                    "estado": "OCUPADO",
                    "agendamiento_id": registro.get("id"),
                    "registrado_en": registro.get("registrado_en"),
                    "google_calendar_estado": registro.get("google_calendar_estado"),
                    "google_event_id": registro.get("google_event_id"),
                    "google_event_link": registro.get("google_event_link"),
                },
            )

    async def _rechazar_horario_ocupado(
        self, comportamiento: CyclicBehaviour, conv_id: str, datos: Dict[str, Any]
    ) -> None:
        try:
            alternativas = self.sugerirHorariosAlternativos(
                str(datos["fecha_hora"]),
                datos.get("alternativas_rechazadas", []),
            )
        except ErrorGoogleCalendar as exc:
            self.registrar_accion(
                "falloConsultaAlternativas()",
                f"error={type(exc).__name__}",
            )
            alternativas = []
        await self.enviar_fipa(
            comportamiento,
            JID_COORDINADOR,
            "REFUSE",
            "(motivo: horario_ocupado)",
            protocolo="FIPA-Query",
            conversation_id=conv_id,
            payload_extra={**datos, "alternativas": alternativas},
        )

    async def _enviar_falla_persistencia(
        self, comportamiento: CyclicBehaviour, conv_id: str, datos: Dict[str, Any]
    ) -> None:
        await self.enviar_fipa(
            comportamiento,
            JID_COORDINADOR,
            "FAILURE",
            "(motivo: persistencia_agendamiento_fallida)",
            protocolo="FIPA-Query",
            conversation_id=conv_id,
            payload_extra=datos,
        )

