"""
Implementacion del AgenteCoordinador (<<play>> rol Coordinador).
Arquitectura: Basado en utilidad.
Relacion <<control>>: Supervisa y delega turnos a EjecutivoVentas,
AnalistaPromociones, LiquidadorCotizaciones y GestorAgenda.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from src.agentes.base_agente import (
    JID_AGENDAMIENTOS,
    JID_COTIZACIONES,
    JID_PROMOCIONES,
    JID_VENTAS,
    AgenteBaseMASML,
)
from src.comunicacion.fipa_acl import extraer_payload_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


@dataclass
class PlanEnrutamiento:
    """Plan MAS-ML para enrutar peticiones desde Ventas hacia especialistas."""

    nombre: str = "PlanEnrutamiento"
    objetivo: str = "Asignar cada caso comercial al especialista correcto con mínima latencia"


@dataclass
class PlanIntermediarPromo:
    """Plan MAS-ML para solicitar, evaluar utilidad e intermediar promociones."""

    nombre: str = "PlanIntermediarPromo"
    objetivo: str = "Maximizar cierre comercial protegiendo el margen mínimo de ganancia"


@dataclass
class PlanEscalar:
    """Plan MAS-ML para suspender automatización y escalar casos hostiles al humano."""

    nombre: str = "PlanEscalar"
    objetivo: str = "Garantizar intervención humana inmediata ante quejas o tono hostil"


class AgenteCoordinador(AgenteBaseMASML):
    """
    <<agent>> AgenteCoordinador:
    Orquestador central basado en utilidad con control de turnos y escalamiento.
    """

    def __init__(
        self,
        entorno: EntornoVirtualAtencion,
        organizacion: SistemaMultiagenteComercial,
        monitor: MonitorFIPAACL,
        jid: str = "coordinador@localhost",
        password: str = "sma_2026",
    ) -> None:
        super().__init__(
            jid=jid,
            password=password,
            nombre_agente="AgenteCoordinador",
            nombre_rol="Coordinador",
            arquitectura_tipo="Basado en utilidad",
            entorno=entorno,
            organizacion=organizacion,
            monitor=monitor,
        )
        self.creencias: Dict[str, Any] = {
            "casosEscalados": [],
            "intencionCliente": None,
            "cotizacionCalculada": None,
            "propuestaPromocion": None,
            "estadoAgentes": {
                "AgenteVentas": "activo",
                "AgenteCotizaciones": "libre",
                "AgentePromociones": "libre",
                "AgenteAgendamientos": "libre",
            },
        }
        self.objetivos = [
            "Coordinar flujo optimo y mantener control humano",
        ]
        self.planes = {
            "PlanEnrutamiento": PlanEnrutamiento(),
            "PlanIntermediarPromo": PlanIntermediarPromo(),
            "PlanEscalar": PlanEscalar(),
        }
        self._contexto_conversaciones: Dict[str, Dict[str, Any]] = {}

    class ComportamientoOrquestador(CyclicBehaviour):
        """Comportamiento SPADE asíncrono que procesa mensajes FIPA-ACL entrantes."""

        async def run(self) -> None:
            msg: Message | None = await self.receive(timeout=0.2)
            if msg is None:
                return
            await self.agent.procesar_mensaje_fipa(self, msg)

    async def setup(self) -> None:
        self.add_behaviour(
            self.ComportamientoOrquestador(), self.crear_plantilla_comercial()
        )

    # ------------------------------------------------------------------
    # ACCIONES DEL DIAGRAMA DE CLASES MAS-ML
    # ------------------------------------------------------------------
    async def enrutarPeticion(
        self,
        comportamiento: CyclicBehaviour,
        tipo_peticion: str,
        payload: Dict[str, Any],
        conversation_id: str,
    ) -> None:
        """Enruta peticiones de AgenteVentas al especialista bajo relación <<control>>."""
        self.creencias["casosEscalados"].append(
            {"conversation_id": conversation_id, "tipo": tipo_peticion, **payload}
        )
        self.creencias["intencionCliente"] = tipo_peticion
        self._contexto_conversaciones[conversation_id] = {
            **self._contexto_conversaciones.get(conversation_id, {}),
            **payload,
        }

        if tipo_peticion == "cotizacion":
            if not self.organizacion.verificar_control("Coordinador", "LiquidadorCotizaciones"):
                raise PermissionError("Relación <<control>> inválida sobre LiquidadorCotizaciones")
            self.registrar_accion("enrutarPeticion()", "caso=cotizacion -> AgenteCotizaciones")

            await self.enviar_fipa(
                comportamiento,
                JID_VENTAS,
                "AGREE",
                "(escalar_caso cotizacion)",
                protocolo="FIPA-Request",
                conversation_id=conversation_id,
                payload_extra={"id_cliente": payload.get("id_cliente", "cliente_001")},
            )
            self.creencias["estadoAgentes"]["AgenteCotizaciones"] = "ocupado"
            await self.enviar_fipa(
                comportamiento,
                JID_COTIZACIONES,
                "REQUEST",
                "(calcular_costo parametros_servicio)",
                protocolo="FIPA-Request",
                conversation_id=conversation_id,
                payload_extra=payload,
            )

        elif tipo_peticion == "evaluar_incentivo":
            await self.solicitarEvaluacionPromo(
                comportamiento=comportamiento,
                payload=payload,
                conversation_id=conversation_id,
            )

        elif tipo_peticion == "agendamiento":
            if not self.organizacion.verificar_control("Coordinador", "GestorAgenda"):
                raise PermissionError("Relación <<control>> inválida sobre GestorAgenda")
            cotizacion_aprobada = bool(
                payload.get("cotizacion_aprobada")
                or self.creencias.get("cotizacionCalculada") is not None
            )
            if not self.organizacion.validar_prerequisito_agendamiento(cotizacion_aprobada):
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "REFUSE",
                    "(motivo: requiere_cotizacion_aprobada)",
                    protocolo="FIPA-Query",
                    conversation_id=conversation_id,
                )
                return

            fecha_hora = str(payload.get("fecha_hora", "2026-11-18 10:00"))
            self.registrar_accion(
                "enrutarPeticion()", f"caso=agendamiento ({fecha_hora}) -> AgenteAgendamientos"
            )
            self.creencias["estadoAgentes"]["AgenteAgendamientos"] = "ocupado"
            await self.enviar_fipa(
                comportamiento,
                JID_AGENDAMIENTOS,
                "QUERY-IF",
                "(disponible_bloque fecha_hora cotizacion_aprobada)",
                protocolo="FIPA-Query",
                conversation_id=conversation_id,
                payload_extra={
                    "fecha_hora": fecha_hora,
                    "cotizacion_aprobada": True,
                    "id_cliente": payload.get("id_cliente", "cliente_001"),
                    "nombre_cliente": payload.get("nombre_cliente", "Cliente Empresarial"),
                },
            )

    async def solicitarEvaluacionPromo(
        self,
        comportamiento: CyclicBehaviour,
        payload: Dict[str, Any],
        conversation_id: str,
    ) -> None:
        """Solicita al AgentePromociones calcular descuento óptimo cuidando margen."""
        if not self.organizacion.verificar_control("Coordinador", "AnalistaPromociones"):
            raise PermissionError("Relación <<control>> inválida sobre AnalistaPromociones")
        self.registrar_accion(
            "solicitarEvaluacionPromo()",
            f"indecision={payload.get('nivel_indecision', 0.72)}",
        )
        self.creencias["estadoAgentes"]["AgentePromociones"] = "ocupado"
        cotizacion_actual = float(
            payload.get("cotizacion_actual")
            or (self.creencias.get("cotizacionCalculada") or {}).get("valor_total", 4_500_000.0)
        )
        await self.enviar_fipa(
            comportamiento,
            JID_PROMOCIONES,
            "REQUEST",
            "(calcular_descuento cotizacion_actual nivel_indecision)",
            protocolo="FIPA-Propose",
            conversation_id=conversation_id,
            payload_extra={
                **payload,
                "cotizacion_actual": cotizacion_actual,
            },
        )

    def evaluar_utilidad_propuesta_promo(
        self, descuento_porciento: float, cotizacion_actual: float
    ) -> Dict[str, Any]:
        """
        Función de utilidad racional del AgenteCoordinador:
        Evalúa si el descuento propuesto protege el margen mínimo de ganancia (22%)
        y respeta el tope máximo permitido (15%), maximizando la conversión esperada.
        """
        bd = self.entorno.tarifasYMargenes
        margen_base = bd.margen_base_estimado
        margen_minimo = bd.margen_minimo_ganancia
        descuento_maximo = bd.descuento_maximo_permitido

        factor_precio = max(0.05, 1.0 - (descuento_porciento / 100.0))
        nuevo_margen = (factor_precio - (1.0 - margen_base)) / factor_precio
        respeta_margen = (descuento_porciento <= descuento_maximo) and (
            nuevo_margen >= margen_minimo
        )
        valor_con_descuento = round(cotizacion_actual * factor_precio, -2)
        utilidad = round(nuevo_margen * 100.0 + (0.4 * descuento_porciento if respeta_margen else -50.0), 2)

        return {
            "acepta": respeta_margen,
            "descuento_porciento": descuento_porciento,
            "nuevo_margen": round(nuevo_margen, 4),
            "margen_minimo": margen_minimo,
            "descuento_maximo": descuento_maximo,
            "valor_con_descuento": valor_con_descuento,
            "utilidad": utilidad,
        }

    async def transferirPromoAVentas(
        self,
        comportamiento: CyclicBehaviour,
        descuento_porciento: float,
        payload: Dict[str, Any],
        conversation_id: str,
    ) -> None:
        """
        Aplica la regla de canal centralizado de promociones entregando la
        oferta aprobada únicamente al AgenteVentas.
        """
        self.registrar_accion(
            "transferirPromoAVentas()",
            f"descuento={descuento_porciento}% -> AgenteVentas",
        )
        etiqueta_pct = f"{int(round(descuento_porciento))}_porciento"
        await self.enviar_fipa(
            comportamiento,
            JID_VENTAS,
            "INFORM",
            f"(transferir_promocion {etiqueta_pct})",
            protocolo="FIPA-Propose",
            conversation_id=conversation_id,
            payload_extra={
                **payload,
                "descuento_porciento": descuento_porciento,
            },
        )

    def evaluarRiesgoYSuspenderSistema(self, motivo_critico: str) -> Dict[str, Any]:
        """Evalúa el riesgo crítico reportado por Ventas y prepara la pausa del sistema."""
        self.registrar_accion(
            "evaluarRiesgoYSuspenderSistema()", f"riesgo=CRÍTICO ({motivo_critico})"
        )
        return {"riesgo": "alto", "suspender_automatizacion": True, "motivo": motivo_critico}

    def pausarYAlertarHumano(
        self, historial_charla: List[Dict[str, Any]], motivo_critico: str, id_cliente: str
    ) -> Dict[str, Any]:
        """Pausa la automatizacion y escala el control al profesional humano en el entorno."""
        self.registrar_accion("pausarYAlertarHumano()", f"cliente={id_cliente}")
        resultado = self.entorno.notificarAlertaHumana(
            historial_charla=historial_charla,
            motivo_critico=motivo_critico,
            id_cliente=id_cliente,
        )
        self.creencias["estadoAgentes"]["AgenteVentas"] = "pausado_por_escalamiento_humano"
        return resultado

    # ------------------------------------------------------------------
    # DESPACHADOR DE MENSAJES FIPA-ACL RECIBIDOS
    # ------------------------------------------------------------------
    async def procesar_mensaje_fipa(
        self, comportamiento: CyclicBehaviour, msg: Message
    ) -> None:
        perf = (msg.get_metadata("performative") or "").upper()
        proto = msg.get_metadata("protocol") or "FIPA-Request"
        conv_id = msg.get_metadata("conversation_id") or msg.thread or "conv-default"
        contenido = msg.body or ""
        payload = extraer_payload_fipa(msg)
        emisor = str(msg.sender).split("@")[0].lower()

        if emisor == "ventas":
            if perf == "REQUEST":
                if "escalar_caso cotizacion" in contenido:
                    await self.enrutarPeticion(comportamiento, "cotizacion", payload, conv_id)
                elif "evaluar_incentivo" in contenido:
                    await self.enrutarPeticion(comportamiento, "evaluar_incentivo", payload, conv_id)
                elif "escalar_caso agendamiento" in contenido:
                    await self.enrutarPeticion(comportamiento, "agendamiento", payload, conv_id)
            elif perf == "INFORM" and "alerta_caso_critico" in contenido:
                motivo = str(payload.get("motivo_critico", "tono_hostil"))
                self._contexto_conversaciones[conv_id] = {**payload, "motivo_critico": motivo}
                self.evaluarRiesgoYSuspenderSistema(motivo)
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "CANCEL",
                    "(cancelar_accion atencion_automatica_chat)",
                    protocolo="FIPA-Request",
                    conversation_id=conv_id,
                    payload_extra={"id_cliente": payload.get("id_cliente", "cliente_001")},
                )
            elif perf == "CONFIRM" and "automatizacion_pausada" in contenido:
                ctx = self._contexto_conversaciones.get(conv_id, {})
                id_cliente = str(payload.get("id_cliente", ctx.get("id_cliente", "cliente_001")))
                canal = str(payload.get("canal", ctx.get("canal", "WhatsApp")))
                motivo_critico = str(ctx.get("motivo_critico", "reclamo_critico_cliente"))
                perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(id_cliente)
                resultado = self.pausarYAlertarHumano(
                    historial_charla=list(perfil.get("historial_interacciones", [])),
                    motivo_critico=motivo_critico,
                    id_cliente=id_cliente,
                )
                self.marcar_ciclo_completado(resultado)

        elif emisor == "cotizaciones":
            if perf == "AGREE":
                self.creencias["estadoAgentes"]["AgenteCotizaciones"] = "procesando"
            elif perf == "INFORM" and "cotizacion_calculada" in contenido:
                self.creencias["cotizacionCalculada"] = payload
                self.creencias["estadoAgentes"]["AgenteCotizaciones"] = "libre"
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "INFORM",
                    "(entregar_cotizacion valor_total desglose)",
                    protocolo="FIPA-Request",
                    conversation_id=conv_id,
                    payload_extra=payload,
                )
            elif perf == "FAILURE":
                self.creencias["estadoAgentes"]["AgenteCotizaciones"] = "libre"
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "REQUEST",
                    "(solicitar_parametros_faltantes cliente)",
                    protocolo="FIPA-Request",
                    conversation_id=conv_id,
                    payload_extra=payload,
                )

        elif emisor == "promociones":
            if perf == "PROPOSE":
                descuento = float(payload.get("descuento_porciento", 10.0))
                cotizacion_actual = float(payload.get("cotizacion_actual", 4_500_000.0))
                evaluacion = self.evaluar_utilidad_propuesta_promo(descuento, cotizacion_actual)
                self.creencias["propuestaPromocion"] = evaluacion
                self._contexto_conversaciones[conv_id] = {
                    **self._contexto_conversaciones.get(conv_id, {}),
                    **payload,
                    **evaluacion,
                }
                evaluacion_con_ctx = {
                    **evaluacion,
                    "id_cliente": payload.get("id_cliente", "cliente_001"),
                    "canal": payload.get("canal", "WhatsApp"),
                }
                if evaluacion["acepta"]:
                    etiqueta_pct = f"{int(round(descuento))}_porciento"
                    await self.enviar_fipa(
                        comportamiento,
                        JID_PROMOCIONES,
                        "ACCEPT-PROPOSAL",
                        f"(aprobar_descuento {etiqueta_pct})",
                        protocolo="FIPA-Propose",
                        conversation_id=conv_id,
                        payload_extra=evaluacion_con_ctx,
                    )
                else:
                    self.creencias["estadoAgentes"]["AgentePromociones"] = "libre"
                    await self.enviar_fipa(
                        comportamiento,
                        JID_PROMOCIONES,
                        "REJECT-PROPOSAL",
                        "(motivo: excede_margen_permitido)",
                        protocolo="FIPA-Propose",
                        conversation_id=conv_id,
                        payload_extra=evaluacion_con_ctx,
                    )
                    await self.enviar_fipa(
                        comportamiento,
                        JID_VENTAS,
                        "REFUSE",
                        "(motivo: descuento_no_viable)",
                        protocolo="FIPA-Propose",
                        conversation_id=conv_id,
                        payload_extra=evaluacion_con_ctx,
                    )
            elif perf == "INFORM" and "promocion_autorizada" in contenido:
                self.creencias["estadoAgentes"]["AgentePromociones"] = "libre"
                ctx = self._contexto_conversaciones.get(conv_id, {})
                descuento = float(payload.get("descuento_porciento", ctx.get("descuento_porciento", 10.0)))
                await self.transferirPromoAVentas(
                    comportamiento=comportamiento,
                    descuento_porciento=descuento,
                    payload={**ctx, **payload},
                    conversation_id=conv_id,
                )

        elif emisor == "agendamientos":
            if perf == "INFORM" and "true" in contenido:
                ctx = self._contexto_conversaciones.get(conv_id, {})
                await self.enviar_fipa(
                    comportamiento,
                    JID_AGENDAMIENTOS,
                    "REQUEST",
                    "(registrar_cita cliente fecha_hora)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra={
                        "fecha_hora": payload.get("fecha_hora", ctx.get("fecha_hora")),
                        "id_cliente": payload.get("id_cliente", ctx.get("id_cliente", "cliente_001")),
                        "nombre_cliente": payload.get("nombre_cliente", ctx.get("nombre_cliente", "Cliente Empresarial")),
                    },
                )
            elif perf == "CONFIRM" and "cita_registrada" in contenido:
                self.creencias["estadoAgentes"]["AgenteAgendamientos"] = "libre"
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "INFORM",
                    "(reserva_confirmada fecha_hora)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra=payload,
                )
            elif perf == "PROPOSE" and "horarios_alternativos" in contenido:
                self.creencias["estadoAgentes"]["AgenteAgendamientos"] = "libre"
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "INFORM",
                    "(ofrecer_alternativas bloque_A bloque_B)",
                    protocolo="FIPA-Query",
                    conversation_id=conv_id,
                    payload_extra=payload,
                )
