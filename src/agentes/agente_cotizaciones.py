"""
Implementacion del AgenteCotizaciones (<<play>> rol LiquidadorCotizaciones).
Las tarifas por sesiones usan el precio comercial configurado; los servicios por
horas conservan el calculo experimental con RandomForestRegressor.
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
from src.ml.modelos_ml import (
    ModeloCotizacionesRF,
    campo_cantidad_para_unidad,
    normalizar_unidad_comercial,
    rango_complejidad_servicio,
)
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
    objetivo: str = "Aplicar la tarifa comercial y presentar un desglose comprensible"


class AgenteCotizaciones(AgenteBaseMASML):
    """
    <<agent>> AgenteCotizaciones:
    Cotiza por sesiones según el catálogo configurado y conserva el modelo
    RandomForest para la estimación de servicios por horas.
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
        parametros = self.normalizarParametrosServicio(parametros_servicio)
        self.creencias["parametrosServicio"] = dict(parametros)
        catalogo = self.entorno.plataformas.portafolio_servicios
        cfg = self.entorno.config_negocio
        etiqueta_comp = str(cfg.get("etiqueta_complejidad", "complejidad"))
        faltantes: List[str] = []

        tipo = parametros.get("tipo_servicio")
        if not tipo or str(tipo) not in catalogo:
            faltantes.append("tipo_servicio")

        unidad = self._unidad_requerida(tipo)
        etiqueta_cantidad = self._etiqueta_cantidad_requerida(tipo, unidad)
        cantidad = self._entero_positivo(parametros.get("cantidad_unidades"))
        if cantidad is None:
            faltantes.append(etiqueta_cantidad)

        requiere_comp = True
        if tipo and str(tipo) in catalogo:
            requiere_comp = bool(catalogo[str(tipo)].get("requiere_complejidad", True))

        complejidad = self._entero_no_negativo(parametros.get("complejidad"))
        rango = rango_complejidad_servicio(
            catalogo.get(str(tipo), {}),
            (
                int(cfg.get("complejidad_minima", 1)),
                int(cfg.get("complejidad_maxima", 5)),
            ),
        )
        complejidad_valida = (
            complejidad is not None and rango[0] <= complejidad <= rango[1]
        )
        if (requiere_comp and not complejidad_valida) or (
            parametros.get("complejidad") is not None and not complejidad_valida
        ):
            faltantes.append(etiqueta_comp)

        self.registrar_accion(
            "extraerRequerimientos()",
            f"valido={len(faltantes) == 0} | faltantes={faltantes}",
        )
        return {"valido": len(faltantes) == 0, "faltantes": faltantes}

    def _unidad_requerida(self, tipo_servicio: Any) -> str:
        catalogo = self.entorno.plataformas.portafolio_servicios
        meta = catalogo.get(str(tipo_servicio), {}) if tipo_servicio else {}
        return str(
            meta.get(
                "unidad_cantidad",
                self.entorno.config_negocio.get("unidad_cantidad", "horas"),
            )
        )

    def _etiqueta_cantidad_requerida(
        self, tipo_servicio: Any, unidad: str
    ) -> str:
        catalogo = self.entorno.plataformas.portafolio_servicios
        meta = catalogo.get(str(tipo_servicio), {}) if tipo_servicio else {}
        if meta.get("etiqueta_cantidad"):
            return str(meta["etiqueta_cantidad"])
        if meta.get("unidad_cantidad"):
            return campo_cantidad_para_unidad(unidad)
        return str(
            self.entorno.config_negocio.get(
                "etiqueta_cantidad", campo_cantidad_para_unidad(unidad)
            )
        )

    def normalizarParametrosServicio(
        self, parametros_servicio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Unifica aliases sin tratar unidades comerciales distintas como equivalentes."""
        parametros = dict(parametros_servicio)
        tipo = parametros.get("tipo_servicio")
        unidad_requerida = self._unidad_requerida(tipo)
        etiqueta = self._etiqueta_cantidad_requerida(tipo, unidad_requerida)
        unidad_recibida = parametros.get("unidad_cantidad")
        if unidad_recibida and (
            normalizar_unidad_comercial(unidad_recibida)
            != normalizar_unidad_comercial(unidad_requerida)
        ):
            parametros.pop("cantidad_unidades", None)
            parametros.pop("horas_estimadas", None)
            parametros.pop("sesiones_requeridas", None)
            parametros.pop(etiqueta, None)
            return parametros

        claves_cantidad = [etiqueta, "cantidad_unidades"]
        if normalizar_unidad_comercial(unidad_requerida) == "hora":
            claves_cantidad.append("horas_estimadas")
        elif normalizar_unidad_comercial(unidad_requerida) == "sesion":
            claves_cantidad.append("sesiones_requeridas")

        cantidad = next(
            (
                self._entero_positivo(parametros.get(clave))
                for clave in dict.fromkeys(claves_cantidad)
                if self._entero_positivo(parametros.get(clave)) is not None
            ),
            None,
        )
        for alias in ("horas_estimadas", "sesiones_requeridas"):
            parametros.pop(alias, None)
        parametros["unidad_cantidad"] = unidad_requerida
        if cantidad is not None:
            parametros["cantidad_unidades"] = cantidad
            parametros[etiqueta] = cantidad
            if normalizar_unidad_comercial(unidad_requerida) == "hora":
                parametros["horas_estimadas"] = cantidad
        else:
            parametros.pop("cantidad_unidades", None)
            parametros.pop(etiqueta, None)
        return parametros

    @staticmethod
    def _entero_positivo(valor: Any) -> int | None:
        if valor is None or isinstance(valor, bool):
            return None
        try:
            numero = int(valor)
        except (TypeError, ValueError):
            return None
        return numero if numero > 0 else None

    @staticmethod
    def _entero_no_negativo(valor: Any) -> int | None:
        if valor is None or isinstance(valor, bool):
            return None
        try:
            return int(valor)
        except (TypeError, ValueError):
            return None

    def ejecutarCalculoCostos(
        self, parametros_servicio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcula el precio comercial por sesiones o estima servicios por horas."""
        tipo_servicio = str(parametros_servicio["tipo_servicio"])
        catalogo = self.entorno.plataformas.portafolio_servicios
        meta = catalogo[tipo_servicio]
        unidad = str(
            meta.get(
                "unidad_cantidad",
                self.entorno.config_negocio.get("unidad_cantidad", "horas"),
            )
        )
        moneda = str(self.entorno.config_negocio.get("moneda", "COP"))

        if normalizar_unidad_comercial(unidad) == "sesion":
            cantidad = self._entero_positivo(
                parametros_servicio.get("cantidad_unidades")
            )
            tarifa_unitaria = meta.get("tarifa_unitaria")
            if cantidad is None:
                raise ValueError("La cotización por sesiones requiere una cantidad válida.")
            if tarifa_unitaria is None:
                raise ValueError(
                    f"El servicio '{tipo_servicio}' no tiene una tarifa unitaria configurada."
                )
            tarifa = float(tarifa_unitaria)
            valor_total = float(tarifa * cantidad)
            complejidad = self._entero_no_negativo(
                parametros_servicio.get("complejidad")
            )
            prediccion = {
                "valor_total": valor_total,
                "desglose": {
                    "servicio": str(meta.get("nombre", tipo_servicio)),
                    "cantidad_unidades": cantidad,
                    "unidad_cantidad": unidad,
                    "tarifa_unitaria": tarifa,
                    "complejidad": complejidad,
                    "urgencia_dias": parametros_servicio.get("urgencia_dias"),
                    "costo_base": valor_total,
                    "ajuste_complejidad_urgencia": 0.0,
                    "adicional_mantenimiento": 0.0,
                },
            }
            self.registrar_accion(
                "ejecutarCalculoCostos()",
                (
                    f"Tarifa configurada por sesiones -> {cantidad} x "
                    f"${tarifa:,.0f} = ${valor_total:,.0f} {moneda}; "
                    f"complejidad={complejidad} sin ajuste tarifario"
                ),
            )
            return prediccion

        prediccion = self.modelo_rf.predecir_cotizacion(parametros_servicio)
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
        contexto_conversacion_id = payload_contexto.get(
            "contexto_conversacion_id"
        )
        parametros = dict(payload_contexto.get("parametros_servicio", {}))
        cantidad_productos = self._entero_positivo(
            parametros.get("cantidad_productos")
        )
        desglose = dict(cotizacion["desglose"])
        if cantidad_productos is not None:
            desglose["cantidad_productos"] = cantidad_productos
        self.entorno.perfilesClientes.registrar_cotizacion_calculada(
            id_cliente,
            valor_total=cotizacion["valor_total"],
            complejidad=int(parametros.get("complejidad", 3)),
            parametros_cotizacion=parametros,
            conversation_id=(
                str(contexto_conversacion_id)
                if contexto_conversacion_id
                else None
            ),
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
                "contexto_conversacion_id": contexto_conversacion_id,
                "valor_total": cotizacion["valor_total"],
                "desglose": desglose,
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
        parametros = self.normalizarParametrosServicio(
            dict(payload.get("parametros_servicio", {}))
        )
        payload["parametros_servicio"] = parametros

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
            self.entorno.perfilesClientes.registrar_parametros_cotizacion_pendientes(
                id_cliente,
                parametros=parametros,
                campos=validacion["faltantes"],
                conversation_id=(
                    str(payload.get("contexto_conversacion_id"))
                    if payload.get("contexto_conversacion_id")
                    else None
                ),
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

