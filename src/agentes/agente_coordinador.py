"""
Implementacion del AgenteCoordinador (<<play>> rol Coordinador).
Arquitectura: Basado en utilidad.
Relacion <<control>>: Supervisa y delega turnos a EjecutivoVentas,
AnalistaPromociones, LiquidadorCotizaciones y GestorAgenda.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
from src.reglas_comerciales import promocion_elegible


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
    def evaluarSolicitudVenta(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evalua alcance y viabilidad operativa de la solicitud de venta."""
        parametros = dict(payload.get("parametros_servicio", {}))
        cantidad = int(
            parametros.get("cantidad_unidades")
            or parametros.get("horas_estimadas")
            or 0
        )
        fuera_alcance = bool(parametros.get("fuera_de_alcance", False))
        viable = (not fuera_alcance) and (cantidad <= 500)
        motivo = "viable" if viable else "fuera_de_alcance_operativo"
        self.registrar_accion(
            "evaluarSolicitudVenta()",
            f"viable={viable} | motivo={motivo}",
        )
        return {"viable": viable, "motivo": motivo}

    def validarPropuestaComercial(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida la propuesta comercial liquidada y protege el margen minimo de
        factibilidad configurado en BD antes de autorizar su entrega al cliente.
        """
        valor_total = float(payload.get("valor_total", 0.0))
        desglose = payload.get("desglose")
        margen_minimo = float(self.entorno.config_negocio.get("margen_minimo", 0.25))
        ajustada = False
        margen_neto = None

        if isinstance(desglose, dict):
            costo_op = float(desglose.get("costo_operativo_directo") or 0.0)
            piso_rentable = float(desglose.get("piso_minimo_rentable") or 0.0)
            if costo_op > 0:
                if piso_rentable <= 0:
                    piso_rentable = round(costo_op / max(0.05, 1.0 - margen_minimo), -3)
                if valor_total < piso_rentable:
                    valor_total = piso_rentable
                    payload["valor_total"] = valor_total
                    desglose["ajustado_por_margen_minimo"] = True
                    ajustada = True
                margen_neto = (
                    round((valor_total - costo_op) / valor_total, 4)
                    if valor_total > 0
                    else 0.0
                )
                desglose["margen_neto_estimado"] = margen_neto
                desglose["validacion_coordinador"] = (
                    f"aprobada_sin_perdida (margen={margen_neto * 100:.1f}% >= min={margen_minimo * 100:.0f}%)"
                )

        detalle = f"valor_total={valor_total:,.0f}"
        if margen_neto is not None:
            detalle += f" | margen={margen_neto * 100:.1f}% >= min={margen_minimo * 100:.0f}% | anti_perdida=OK"
            if ajustada:
                detalle += " (ajustado_al_piso_rentable)"
        self.registrar_accion("validarPropuestaComercial()", detalle)
        return {
            "valida": valor_total >= 0.0,
            "valor_total": valor_total,
            "margen_neto": margen_neto,
            "ajustada_por_margen": ajustada,
        }

    def cerrarOportunidadVenta(
        self,
        id_cliente: str,
        motivo: str,
        estado: str = "descartada",
        contexto_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Registra el cierre de oportunidad cuando el cliente declina la oferta."""
        self.registrar_accion(
            f"cerrarOportunidadVenta(estado={estado})",
            f"motivo={motivo}",
        )
        self.entorno.perfilesClientes.actualizar_estado_conversacion(
            id_cliente,
            contexto_id,
            {
                "esperando_parametros_cotizacion": False,
                "estado_reserva": {},
                "estado_oportunidad": estado,
            },
        )
        self.entorno.perfilesClientes.actualizar_perfil(
            id_cliente,
            esperando_parametros_cotizacion=False,
            estado_oportunidad=estado,
        )
        cierre = {"id_cliente": id_cliente, "motivo": motivo, "estado": estado}
        self.creencias["ultimoCierreOportunidad"] = cierre
        return cierre

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
            eval_venta = self.evaluarSolicitudVenta(payload)
            if not eval_venta["viable"]:
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "REFUSE",
                    "(motivo: fuera_de_alcance_operativo)",
                    protocolo="FIPA-Request",
                    conversation_id=conversation_id,
                    payload_extra={
                        "id_cliente": payload.get("id_cliente", "cliente_001"),
                        "canal": payload.get("canal", "WhatsApp"),
                        "contexto_conversacion_id": payload.get("contexto_conversacion_id"),
                        "motivo": eval_venta["motivo"],
                    },
                )
                return

            self.registrar_accion("enrutarPeticion()", "caso=cotizacion -> AgenteCotizaciones")

            await self.enviar_fipa(
                comportamiento,
                JID_VENTAS,
                "AGREE",
                "(solicitar_cotizacion_servicio)",
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
            id_cliente = str(payload.get("id_cliente", "cliente_001"))
            contexto_id = payload.get("contexto_conversacion_id")
            cotizacion_aprobada = (
                payload.get("cotizacion_aprobada") is True
                and self.entorno.perfilesClientes.cotizacion_aprobada(
                    id_cliente, str(contexto_id) if contexto_id else None
                )
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

            fecha_hora = str(payload.get("fecha_hora", "")).strip()
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
                    "cotizacion_aprobada": cotizacion_aprobada,
                    "id_cliente": id_cliente,
                    "nombre_cliente": payload.get("nombre_cliente", "Cliente Empresarial"),
                    "canal": payload.get("canal", "WhatsApp"),
                    "servicio": payload.get("servicio"),
                    "contexto_conversacion_id": contexto_id,
                    "alternativas_rechazadas": payload.get(
                        "alternativas_rechazadas", []
                    ),
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
        id_cliente = str(payload.get("id_cliente", ""))
        contexto_id = payload.get("contexto_conversacion_id")
        try:
            cotizacion_actual = float(payload.get("cotizacion_actual"))
        except (TypeError, ValueError):
            cotizacion_actual = 0.0
        codigo = str(payload.get("promocion_codigo", "")).strip()
        if (
            cotizacion_actual <= 0
            or not self.entorno.perfilesClientes.tiene_cotizacion(
                id_cliente, str(contexto_id) if contexto_id else None
            )
            or not codigo
        ):
            await self.enviar_fipa(
                comportamiento,
                JID_VENTAS,
                "REFUSE",
                "(motivo: descuento_no_viable_sin_cotizacion_o_promocion)",
                protocolo="FIPA-Propose",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": payload.get("canal", "WhatsApp"),
                    "contexto_conversacion_id": contexto_id,
                },
            )
            return
        self.registrar_accion(
            "solicitarEvaluacionPromo()",
            f"indecision={payload.get('nivel_indecision', 0.72)}",
        )
        self.creencias["estadoAgentes"]["AgentePromociones"] = "ocupado"
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
        self,
        descuento_porciento: float,
        cotizacion_actual: float,
        descuento_maximo_autorizado: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Función de utilidad racional del AgenteCoordinador:
        Evalúa si el descuento propuesto protege el margen mínimo de ganancia (22%)
        y respeta el tope máximo permitido (15%), maximizando la conversión esperada.
        """
        bd = self.entorno.tarifasYMargenes
        margen_base = bd.margen_base_estimado
        margen_minimo = bd.margen_minimo_ganancia
        descuento_maximo = min(
            bd.descuento_maximo_permitido,
            descuento_maximo_autorizado
            if descuento_maximo_autorizado is not None
            else bd.descuento_maximo_permitido,
        )

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
            "evaluarRiesgoYSuspenderSistema()", "riesgo=CRITICO"
        )
        return {"riesgo": "alto", "suspender_automatizacion": True, "motivo": motivo_critico}

    def pausarYAlertarHumano(
        self, historial_charla: List[Dict[str, Any]], motivo_critico: str, id_cliente: str
    ) -> Dict[str, Any]:
        """Pausa la automatizacion y escala el control al profesional humano en el entorno."""
        self.registrar_accion("pausarYAlertarHumano()")
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
                if (
                    "solicitar_cotizacion_servicio" in contenido
                    or "escalar_caso cotizacion" in contenido
                ):
                    await self.enrutarPeticion(comportamiento, "cotizacion", payload, conv_id)
                elif "evaluar_incentivo" in contenido:
                    await self.enrutarPeticion(comportamiento, "evaluar_incentivo", payload, conv_id)
                elif (
                    "solicitar_agendamiento" in contenido
                    or "escalar_caso agendamiento" in contenido
                ):
                    await self.enrutarPeticion(comportamiento, "agendamiento", payload, conv_id)
            elif perf == "INFORM" and "parametros_completados" in contenido:
                self._contexto_conversaciones[conv_id] = {
                    **self._contexto_conversaciones.get(conv_id, {}),
                    **payload,
                }
                self.creencias["estadoAgentes"]["AgenteCotizaciones"] = "ocupado"
                await self.enviar_fipa(
                    comportamiento,
                    JID_COTIZACIONES,
                    "INFORM",
                    "(actualizar_parametros parametros_servicio)",
                    protocolo="FIPA-Request",
                    conversation_id=conv_id,
                    payload_extra=payload,
                )
            elif perf == "INFORM" and "cliente_declina_oferta" in contenido:
                id_cliente = str(payload.get("id_cliente", "cliente_001"))
                motivo = str(payload.get("motivo_rechazo", "excede_presupuesto"))
                contexto_id_raw = payload.get("contexto_conversacion_id")
                contexto_id = str(contexto_id_raw) if contexto_id_raw else None
                self.cerrarOportunidadVenta(
                    id_cliente=id_cliente,
                    motivo=motivo,
                    estado="descartada",
                    contexto_id=contexto_id,
                )
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "INFORM",
                    "(confirmar_cierre_conversacion)",
                    protocolo="FIPA-Request",
                    conversation_id=conv_id,
                    payload_extra={
                        "id_cliente": id_cliente,
                        "canal": payload.get("canal", "WhatsApp"),
                        "contexto_conversacion_id": contexto_id,
                        "estado": "descartada",
                    },
                )
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
                self.validarPropuestaComercial(payload)
                self.creencias["cotizacionCalculada"] = payload
                self.creencias["estadoAgentes"]["AgenteCotizaciones"] = "libre"
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "INFORM",
                    "(entregar_cotizacion_formal valor_total desglose)",
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
            if perf == "AGREE":
                self.creencias["estadoAgentes"]["AgentePromociones"] = "procesando"
            elif perf == "PROPOSE":
                id_cliente = str(payload.get("id_cliente", ""))
                contexto_id = payload.get("contexto_conversacion_id")
                codigo_promocion = str(payload.get("promocion_codigo", "")).strip()
                promocion = next(
                    (
                        item
                        for item in self.entorno.tarifasYMargenes.promociones_vigentes
                        if str(item.get("codigo", "")) == codigo_promocion
                    ),
                    None,
                )
                cantidad = payload.get(
                    "cantidad_unidades", payload.get("cantidad")
                )
                servicio = str(payload.get("servicio", ""))
                try:
                    cotizacion_actual = float(payload.get("cotizacion_actual"))
                    descuento_propuesto = float(payload.get("descuento_porciento"))
                    cantidad_entera = int(cantidad)
                    complejidad = (
                        int(payload["complejidad"])
                        if payload.get("complejidad") is not None
                        else None
                    )
                except (TypeError, ValueError):
                    promocion = None
                    cotizacion_actual = 0.0
                    descuento_propuesto = 0.0
                    cantidad_entera = 0
                    complejidad = None

                cotizacion_vigente = self.entorno.perfilesClientes.tiene_cotizacion(
                    id_cliente, str(contexto_id) if contexto_id else None
                )
                cotizacion_aprobada = self.entorno.perfilesClientes.cotizacion_aprobada(
                    id_cliente, str(contexto_id) if contexto_id else None
                )
                if (
                    not cotizacion_vigente
                    or cotizacion_aprobada
                    or cotizacion_actual <= 0
                    or descuento_propuesto <= 0
                    or not promocion
                    or not promocion_elegible(
                        promocion,
                        servicio,
                        cantidad_entera,
                        complejidad,
                        canal=str(payload.get("canal", "")),
                    )
                ):
                    self.creencias["estadoAgentes"]["AgentePromociones"] = "libre"
                    await self.enviar_fipa(
                        comportamiento,
                        JID_PROMOCIONES,
                        "REJECT-PROPOSAL",
                        "(motivo: regla_comercial_no_autoriza)",
                        protocolo="FIPA-Propose",
                        conversation_id=conv_id,
                        payload_extra=payload,
                    )
                    await self.enviar_fipa(
                        comportamiento,
                        JID_VENTAS,
                        "REFUSE",
                        "(motivo: descuento_no_viable)",
                        protocolo="FIPA-Propose",
                        conversation_id=conv_id,
                        payload_extra={
                            "id_cliente": id_cliente,
                            "canal": payload.get("canal", "WhatsApp"),
                            "contexto_conversacion_id": contexto_id,
                        },
                    )
                    return

                limite_promocion = min(
                    float(promocion["descuento_max"]),
                    self.entorno.tarifasYMargenes.descuento_maximo_permitido,
                )
                descuento = min(descuento_propuesto, limite_promocion)
                evaluacion = self.evaluar_utilidad_propuesta_promo(
                    descuento,
                    cotizacion_actual,
                    limite_promocion,
                )
                self.creencias["propuestaPromocion"] = evaluacion
                self._contexto_conversaciones[conv_id] = {
                    **self._contexto_conversaciones.get(conv_id, {}),
                    **payload,
                    **evaluacion,
                    "promocion_codigo": codigo_promocion,
                }
                evaluacion_con_ctx = {
                    **evaluacion,
                    "id_cliente": payload.get("id_cliente", "cliente_001"),
                    "canal": payload.get("canal", "WhatsApp"),
                    "contexto_conversacion_id": contexto_id,
                    "promocion_codigo": codigo_promocion,
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
            elif perf == "REFUSE":
                self.creencias["estadoAgentes"]["AgentePromociones"] = "libre"
                await self.enviar_fipa(
                    comportamiento,
                    JID_VENTAS,
                    "REFUSE",
                    "(motivo: descuento_no_viable)",
                    protocolo="FIPA-Propose",
                    conversation_id=conv_id,
                    payload_extra={
                        "id_cliente": payload.get("id_cliente", "cliente_001"),
                        "canal": payload.get("canal", "WhatsApp"),
                        "contexto_conversacion_id": payload.get(
                            "contexto_conversacion_id"
                        ),
                    },
                )
            elif perf == "INFORM" and "promocion_autorizada" in contenido:
                self.creencias["estadoAgentes"]["AgentePromociones"] = "libre"
                ctx = self._contexto_conversaciones.get(conv_id, {})
                descuento = float(
                    payload.get(
                        "descuento_porciento",
                        ctx.get("descuento_porciento", 0.0),
                    )
                )
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
                        "canal": payload.get("canal", ctx.get("canal", "WhatsApp")),
                        "servicio": payload.get("servicio", ctx.get("servicio")),
                        "cotizacion_aprobada": payload.get(
                            "cotizacion_aprobada",
                            ctx.get("cotizacion_aprobada", False),
                        ),
                        "contexto_conversacion_id": payload.get(
                            "contexto_conversacion_id",
                            ctx.get("contexto_conversacion_id"),
                        ),
                        "alternativas_rechazadas": payload.get(
                            "alternativas_rechazadas",
                            ctx.get("alternativas_rechazadas", []),
                        ),
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
            elif perf == "REFUSE":
                self.creencias["estadoAgentes"]["AgenteAgendamientos"] = "libre"
                if "horario_ocupado" in contenido:
                    await self.enviar_fipa(
                        comportamiento,
                        JID_VENTAS,
                        "INFORM",
                        "(ofrecer_alternativas bloque_A bloque_B)",
                        protocolo="FIPA-Query",
                        conversation_id=conv_id,
                        payload_extra=payload,
                    )
                else:
                    await self.enviar_fipa(
                        comportamiento,
                        JID_VENTAS,
                        "REFUSE",
                        "(error_agendamiento datos_rechazados)",
                        protocolo="FIPA-Query",
                        conversation_id=conv_id,
                        payload_extra=payload,
                    )
            elif perf == "FAILURE":
                self.creencias["estadoAgentes"]["AgenteAgendamientos"] = "libre"
                if "horario_ocupado" in contenido:
                    await self.enviar_fipa(
                        comportamiento,
                        JID_VENTAS,
                        "INFORM",
                        "(ofrecer_alternativas bloque_A bloque_B)",
                        protocolo="FIPA-Query",
                        conversation_id=conv_id,
                        payload_extra=payload,
                    )
                else:
                    await self.enviar_fipa(
                        comportamiento,
                        JID_VENTAS,
                        "FAILURE",
                        "(error_agendamiento persistencia_o_disponibilidad)",
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
