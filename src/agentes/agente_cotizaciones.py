"""
Implementacion del AgenteCotizaciones (<<play>> rol LiquidadorCotizaciones).
Arquitectura: Con aprendizaje (RandomForestRegressor de scikit-learn entrenado
con el historico de tarifas del entorno) y basado en metas.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
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
from src.ml.modelos_ml import ModeloCotizacionesRF
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


@dataclass
class PlanExtraccionDatos:
    """Plan MAS-ML para extraer y validar los parámetros obligatorios del servicio."""

    nombre: str = "PlanExtraccionDatos"
    objetivo: str = "Garantizar que existan variables obligatorias antes de liquidar"


@dataclass
class PlanCalculoTarifa:
    """Plan MAS-ML para calcular costos mediante RandomForestRegressor."""

    nombre: str = "PlanCalculoTarifa"
    objetivo: str = "Dar valoración económica precisa y comprensible con desglose"


class AgenteCotizaciones(AgenteBaseMASML):
    """
    <<agent>> AgenteCotizaciones:
    Liquida presupuestos formales usando RandomForestRegressor sobre el
    histórico de cotizaciones de EntornoVirtualAtencion.
    """

    PARAMETROS_OBLIGATORIOS = ("tipo_servicio", "horas_estimadas")

    def __init__(
        self,
        entorno: EntornoVirtualAtencion,
        organizacion: SistemaMultiagenteComercial,
        monitor: MonitorFIPAACL,
        modelo_rf: ModeloCotizacionesRF,
        jid: str = "cotizaciones@localhost",
        password: str = "sma_2026",
    ) -> None:
        super().__init__(
            jid=jid,
            password=password,
            nombre_agente="AgenteCotizaciones",
            nombre_rol="LiquidadorCotizaciones",
            arquitectura_tipo="Con aprendizaje (RandomForestRegressor) y basado en metas",
            entorno=entorno,
            organizacion=organizacion,
            monitor=monitor,
        )
        self.modelo_rf = modelo_rf
        self.creencias: Dict[str, Any] = {
            "parametrosServicio": {},
            "cotizacionesHistoricas": len(entorno.tarifasYMargenes.tarifas_historicas),
        }
        self.objetivos = [
            "Dar valoracion economica precisa y comprensible",
        ]
        self.planes = {
            "PlanExtraccionDatos": PlanExtraccionDatos(),
            "PlanCalculoTarifa": PlanCalculoTarifa(),
        }

    class ComportamientoLiquidador(CyclicBehaviour):
        """Escucha solicitudes REQUEST de cálculo de costos desde Coordinador."""

        async def run(self) -> None:
            msg: Message | None = await self.receive(timeout=0.2)
            if msg is None:
                return
            await self.agent.procesar_solicitud_cotizacion(self, msg)

    async def setup(self) -> None:
        self.add_behaviour(
            self.ComportamientoLiquidador(), self.crear_plantilla_comercial()
        )

    # ------------------------------------------------------------------
    # ACCIONES DEL DIAGRAMA DE CLASES MAS-ML
    # ------------------------------------------------------------------
    def extraerRequerimientos(
        self, parametros_servicio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extrae y valida los parametros obligatorios segun el catalogo y negocio en BD."""
        self.creencias["parametrosServicio"] = dict(parametros_servicio)
        catalogo = self.entorno.plataformas.portafolio_servicios
        cfg = self.entorno.config_negocio
        etiqueta_cantidad = str(cfg.get("etiqueta_cantidad", "horas_estimadas"))
        etiqueta_comp = str(cfg.get("etiqueta_complejidad", "complejidad"))
        faltantes: List[str] = []

        tipo = parametros_servicio.get("tipo_servicio")
        if not tipo or str(tipo) not in catalogo:
            faltantes.append("tipo_servicio")

        cantidad = (
            parametros_servicio.get("cantidad_unidades")
            or parametros_servicio.get("horas_estimadas")
        )
        if cantidad is None or int(cantidad) <= 0:
            faltantes.append(etiqueta_cantidad)

        requiere_comp = True
        if tipo and str(tipo) in catalogo:
            requiere_comp = bool(catalogo[str(tipo)].get("requiere_complejidad", True))

        complejidad = parametros_servicio.get("complejidad")
        if requiere_comp and complejidad is None:
            faltantes.append(etiqueta_comp)

        self.registrar_accion(
            "extraerRequerimientos()",
            f"valido={len(faltantes) == 0} | faltantes={faltantes}",
        )
        return {"valido": len(faltantes) == 0, "faltantes": faltantes}

    def ejecutarCalculoCostos(
        self, parametros_servicio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta el calculo economico con RandomForestRegressor de scikit-learn."""
        prediccion = self.modelo_rf.predecir_cotizacion(parametros_servicio)
        moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
        self.registrar_accion(
            "ejecutarCalculoCostos()",
            f"RandomForest -> ${prediccion['valor_total']:,.0f} {moneda} (R2={prediccion['desglose']['r2_modelo_rf']})",
        )
        return prediccion

    async def enviarCotizacionFormal(
        self,
        comportamiento: CyclicBehaviour,
        cotizacion: Dict[str, Any],
        payload_contexto: Dict[str, Any],
        conversation_id: str,
    ) -> None:
        """Envia al Coordinador el mensaje INFORM con el valor total y desglose."""
        moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
        self.registrar_accion(
            "enviarCotizacionFormal()",
            f"total=${cotizacion['valor_total']:,.0f} {moneda}",
        )
        id_cliente = str(payload_contexto.get("id_cliente", "cliente_001"))
        parametros = dict(payload_contexto.get("parametros_servicio", {}))
        self.entorno.perfilesClientes.actualizar_perfil(
            id_cliente,
            ultima_cotizacion=cotizacion["valor_total"],
            ultima_complejidad=int(parametros.get("complejidad", 3)),
            parametros_parciales={},
            esperando_parametros_cotizacion=False,
        )
        await self.enviar_fipa(
            comportamiento,
            JID_COORDINADOR,
            "INFORM",
            "(cotizacion_calculada valor_total desglose)",
            protocolo="FIPA-Request",
            conversation_id=conversation_id,
            payload_extra={
                "id_cliente": id_cliente,
                "canal": payload_contexto.get("canal", "WhatsApp"),
                "valor_total": cotizacion["valor_total"],
                "desglose": cotizacion["desglose"],
            },
        )

    async def procesar_solicitud_cotizacion(
        self, comportamiento: CyclicBehaviour, msg: Message
    ) -> None:
        perf = (msg.get_metadata("performative") or "").upper()
        conv_id = msg.get_metadata("conversation_id") or msg.thread or "conv-default"
        if perf != "REQUEST":
            return

        payload = extraer_payload_fipa(msg)
        parametros = dict(payload.get("parametros_servicio", {}))

        # Respuesta inmediata AGREE según Diagrama de Secuencia 1
        await self.enviar_fipa(
            comportamiento,
            JID_COORDINADOR,
            "AGREE",
            "(calcular_costo parametros_servicio)",
            protocolo="FIPA-Request",
            conversation_id=conv_id,
            payload_extra={"id_cliente": payload.get("id_cliente", "cliente_001")},
        )

        # Consulta histórico al entorno (datos(tarifas_historicas))
        referencias = self.entorno.consultarTarifasHistoricas(parametros)
        self.creencias["cotizacionesHistoricas"] = referencias["registros"]

        validacion = self.extraerRequerimientos(parametros)
        if not validacion["valido"]:
            id_cliente = str(payload.get("id_cliente", "cliente_001"))
            self.entorno.perfilesClientes.actualizar_perfil(
                id_cliente,
                parametros_parciales=parametros,
                esperando_parametros_cotizacion=True,
            )
            # Rama else [Faltan parametros obligatorios] -> FAILURE
            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "FAILURE",
                "(motivo: parametros_incompletos)",
                protocolo="FIPA-Request",
                conversation_id=conv_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": payload.get("canal", "WhatsApp"),
                    "faltantes": validacion["faltantes"],
                },
            )
            return

        # Rama alt [Calculo exitoso] -> RandomForestRegressor + INFORM
        cotizacion = self.ejecutarCalculoCostos(parametros)
        await self.enviarCotizacionFormal(
            comportamiento=comportamiento,
            cotizacion=cotizacion,
            payload_contexto=payload,
            conversation_id=conv_id,
        )

