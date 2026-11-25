"""
Implementacion del AgentePromociones (<<play>> rol AnalistaPromociones).
Arquitectura: Con aprendizaje (KNeighborsRegressor - Regresion KNN de scikit-learn)
y basado en utilidad.
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
from src.ml.modelos_ml import ModeloPromocionesKNN
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.reglas_comerciales import promocion_elegible


@dataclass
class PlanEvaluarFactibilidad:
    """Plan MAS-ML para validar promociones vigentes y topes de rentabilidad."""

    nombre: str = "PlanEvaluarFactibilidad"
    objetivo: str = "Verificar catálogo de promociones activas y márgenes históricos"


@dataclass
class PlanSeleccionarDescuento:
    """Plan MAS-ML para estimar descuento óptimo con KNeighborsRegressor."""

    nombre: str = "PlanSeleccionarDescuento"
    objetivo: str = "Maximizar cierre cediendo el menor margen posible"


class AgentePromociones(AgenteBaseMASML):
    """
    <<agent>> AgentePromociones:
    Evalúa incentivos de retención mediante Regresión KNN y envía propuestas
    únicamente al AgenteCoordinador (canal centralizado de promociones).
    """

    def __init__(
        self,
        entorno: EntornoVirtualAtencion,
        organizacion: SistemaMultiagenteComercial,
        monitor: MonitorFIPAACL,
        modelo_knn: ModeloPromocionesKNN,
        jid: str = "promociones@localhost",
        password: str = "sma_2026",
    ) -> None:
        super().__init__(
            jid=jid,
            password=password,
            nombre_agente="AgentePromociones",
            nombre_rol="AnalistaPromociones",
            arquitectura_tipo="Con aprendizaje (KNeighborsRegressor KNN) y basado en utilidad",
            entorno=entorno,
            organizacion=organizacion,
            monitor=monitor,
        )
        self.modelo_knn = modelo_knn
        self.creencias: Dict[str, Any] = {
            "solicitudCoordinador": None,
            "nivelIndecision": 0.0,
            "reglasNegocioYMargenes": {
                "margen_minimo": entorno.tarifasYMargenes.margen_minimo_ganancia,
                "descuento_maximo": entorno.tarifasYMargenes.descuento_maximo_permitido,
            },
            "historicoOfertas": len(entorno.tarifasYMargenes.historico_ofertas),
        }
        self.objetivos = [
            "Maximizar cierre cediendo el menor margen posible",
        ]
        self.planes = {
            "PlanEvaluarFactibilidad": PlanEvaluarFactibilidad(),
            "PlanSeleccionarDescuento": PlanSeleccionarDescuento(),
        }
        self._propuestas_pendientes: Dict[str, Dict[str, Any]] = {}

    class ComportamientoAnalistaPromo(CyclicBehaviour):
        """Atiende REQUEST y decisiones ACCEPT-PROPOSAL / REJECT-PROPOSAL."""

        async def run(self) -> None:
            msg: Message | None = await self.receive(timeout=0.2)
            if msg is None:
                return
            await self.agent.procesar_mensaje_promo(self, msg)

    async def setup(self) -> None:
        self.add_behaviour(
            self.ComportamientoAnalistaPromo(), self.crear_plantilla_comercial()
        )

    # ------------------------------------------------------------------
    # ACCIONES DEL DIAGRAMA DE CLASES MAS-ML
    # ------------------------------------------------------------------
    def validarPromocionesVigentes(self, datos_entorno: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Valida promociones activas y reglas de negocio del entorno."""
        vigentes = [
            p for p in datos_entorno.get("promociones_vigentes", []) if p.get("activa", False)
        ]
        self.registrar_accion(
            "validarPromocionesVigentes()",
            f"activas={len(vigentes)} | margen_min={datos_entorno.get('margen_minimo', 0.22):.0%}",
        )
        return vigentes

    def calcularMargenDescuento(
        self,
        cotizacion_actual: float,
        nivel_indecision: float,
        complejidad: int = 3,
        fidelidad_cliente: int = 1,
        descuento_solicitado: float = 0.0,
        descuento_maximo_autorizado: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Predice el porcentaje de descuento optimo usando KNeighborsRegressor (KNN)
        de scikit-learn sobre el historico de ofertas. Si el cliente exige un
        porcentaje especifico elevado, evalua la propuesta resultante frente a
        los limites de rentabilidad.
        """
        pred = self.modelo_knn.predecir_descuento_optimo(
            monto_cotizado=cotizacion_actual,
            nivel_indecision=nivel_indecision,
            complejidad=complejidad,
            fidelidad_cliente=fidelidad_cliente,
        )
        descuento_knn = float(pred["descuento_porciento"])
        descuento_recomendado = (
            max(descuento_knn, descuento_solicitado)
            if descuento_solicitado > 0.0
            else descuento_knn
        )
        descuento_pct = min(
            max(descuento_recomendado, 0.0),
            max(descuento_maximo_autorizado, 0.0),
        )
        margen_base = self.entorno.tarifasYMargenes.margen_base_estimado
        factor = max(0.05, 1.0 - descuento_pct / 100.0)
        margen_proyectado = round((factor - (1.0 - margen_base)) / factor, 4)

        self.registrar_accion(
            "calcularMargenDescuento()",
            f"KNN={descuento_pct:.1f}% | margen_proyectado={margen_proyectado:.1%} (R2={pred['r2_modelo_knn']})",
        )
        return {
            **pred,
            "descuento_porciento": descuento_pct,
            "margen_proyectado": margen_proyectado,
            "cotizacion_actual": cotizacion_actual,
        }

    async def emitirPropuestaACoordinador(
        self,
        comportamiento: CyclicBehaviour,
        propuesta: Dict[str, Any],
        payload_contexto: Dict[str, Any],
        conversation_id: str,
    ) -> None:
        """
        Emite PROPOSE al AgenteCoordinador respetando la regla de canal
        centralizado de promociones.
        """
        if not self.organizacion.aplicar_regla_canal_centralizado_promociones(
            "AnalistaPromociones", "Coordinador"
        ):
            raise PermissionError("Violación del canal centralizado de promociones")

        descuento_pct = float(propuesta["descuento_porciento"])
        etiqueta_pct = f"{int(round(descuento_pct))}_porciento"
        self.registrar_accion("emitirPropuestaACoordinador()", f"PROPOSE {etiqueta_pct}")
        self._propuestas_pendientes[conversation_id] = propuesta

        await self.enviar_fipa(
            comportamiento,
            JID_COORDINADOR,
            "PROPOSE",
            f"(aplicar_descuento {etiqueta_pct})",
            protocolo="FIPA-Propose",
            conversation_id=conversation_id,
            payload_extra={
                **payload_contexto,
                "descuento_porciento": descuento_pct,
                "margen_proyectado": propuesta["margen_proyectado"],
                "vecinos_knn": propuesta["vecinos_cercanos"],
            },
        )

    async def procesar_mensaje_promo(
        self, comportamiento: CyclicBehaviour, msg: Message
    ) -> None:
        perf = (msg.get_metadata("performative") or "").upper()
        conv_id = msg.get_metadata("conversation_id") or msg.thread or "conv-default"
        payload = extraer_payload_fipa(msg)

        if perf == "REQUEST":
            id_cliente = str(payload.get("id_cliente", ""))
            contexto_id = payload.get("contexto_conversacion_id")
            try:
                cotizacion_actual = float(payload.get("cotizacion_actual"))
                cantidad = int(
                    payload.get("cantidad_unidades", payload.get("cantidad"))
                )
                complejidad = (
                    int(payload["complejidad"])
                    if payload.get("complejidad") is not None
                    else None
                )
            except (TypeError, ValueError):
                cotizacion_actual = 0.0
                cantidad = 0
                complejidad = None
            codigo = str(payload.get("promocion_codigo", "")).strip()
            promocion = next(
                (
                    item
                    for item in self.entorno.tarifasYMargenes.promociones_vigentes
                    if str(item.get("codigo", "")) == codigo
                ),
                None,
            )
            cotizacion_vigente = self.entorno.perfilesClientes.tiene_cotizacion(
                id_cliente, str(contexto_id) if contexto_id else None
            )
            cotizacion_aprobada = self.entorno.perfilesClientes.cotizacion_aprobada(
                id_cliente, str(contexto_id) if contexto_id else None
            )
            if (
                cotizacion_actual <= 0
                or not cotizacion_vigente
                or cotizacion_aprobada
                or not promocion
                or not promocion_elegible(
                    promocion,
                    str(payload.get("servicio", "")),
                    cantidad,
                    complejidad,
                    canal=str(payload.get("canal", "")),
                )
            ):
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: regla_comercial_no_autoriza)",
                    protocolo="FIPA-Propose",
                    conversation_id=conv_id,
                    payload_extra={
                        "id_cliente": id_cliente,
                        "canal": payload.get("canal", "WhatsApp"),
                        "contexto_conversacion_id": contexto_id,
                    },
                )
                return

            nivel_indecision = float(payload.get("nivel_indecision", 0.72))
            complejidad = int(payload.get("complejidad") or 3)
            fidelidad = int(payload.get("fidelidad_cliente", 1))
            descuento_solicitado = float(payload.get("descuento_solicitado_cliente", 0.0))

            self.creencias["solicitudCoordinador"] = payload
            self.creencias["nivelIndecision"] = nivel_indecision

            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "AGREE",
                "(calcular_descuento cotizacion_actual)",
                protocolo="FIPA-Propose",
                conversation_id=conv_id,
                payload_extra={"id_cliente": id_cliente},
            )

            datos_entorno = self.entorno.consultarHistoricoYMargenes(cotizacion_actual)
            self.validarPromocionesVigentes(datos_entorno)
            limite_promocion = min(
                float(promocion["descuento_max"]),
                self.entorno.tarifasYMargenes.descuento_maximo_permitido,
            )
            propuesta = self.calcularMargenDescuento(
                cotizacion_actual=cotizacion_actual,
                nivel_indecision=nivel_indecision,
                complejidad=complejidad,
                fidelidad_cliente=fidelidad,
                descuento_solicitado=descuento_solicitado,
                descuento_maximo_autorizado=limite_promocion,
            )
            if propuesta["descuento_porciento"] <= 0:
                await self.enviar_fipa(
                    comportamiento,
                    JID_COORDINADOR,
                    "REFUSE",
                    "(motivo: descuento_no_autorizado)",
                    protocolo="FIPA-Propose",
                    conversation_id=conv_id,
                    payload_extra={
                        "id_cliente": id_cliente,
                        "canal": payload.get("canal", "WhatsApp"),
                        "contexto_conversacion_id": contexto_id,
                    },
                )
                return
            await self.emitirPropuestaACoordinador(
                comportamiento=comportamiento,
                propuesta=propuesta,
                payload_contexto=payload,
                conversation_id=conv_id,
            )

        elif perf == "ACCEPT-PROPOSAL":
            prop = self._propuestas_pendientes.get(conv_id, {})
            descuento = float(payload.get("descuento_porciento", prop.get("descuento_porciento", 10.0)))
            etiqueta_pct = f"{int(round(descuento))}_porciento"
            self.registrar_accion("confirmarAutorizacionPromo()", f"ACCEPT-PROPOSAL -> {etiqueta_pct}")
            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "INFORM",
                f"(promocion_autorizada {etiqueta_pct})",
                protocolo="FIPA-Propose",
                conversation_id=conv_id,
                payload_extra={
                    "descuento_porciento": descuento,
                    "valor_con_descuento": payload.get("valor_con_descuento"),
                },
            )

        elif perf == "REJECT-PROPOSAL":
            self.registrar_accion(
                "registrarRechazoRentabilidad()",
                "Coordinador rechazó propuesta por margen mínimo",
            )
            self._propuestas_pendientes.pop(conv_id, None)

