"""
Implementacion del entorno compartido <<environment>> EntornoVirtualAtencion
segun el Diagrama de Clases MAS-ML y la Tabla de Analisis del Entorno,
integrado con Botpress (WhatsApp, Webchat, Messenger e Instagram).

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.ml.datasets import CATALOGO_SERVICIOS


@dataclass
class RedesSociales:
    """Representa el atributo `plataformas : RedesSociales` del entorno."""

    canales_activos: List[str] = field(
        default_factory=lambda: ["WhatsApp", "Instagram", "Facebook"]
    )
    portafolio_servicios: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: dict(CATALOGO_SERVICIOS)
    )
    bandeja_entrada: List[Dict[str, Any]] = field(default_factory=list)
    bandeja_salida: List[Dict[str, Any]] = field(default_factory=list)

    def recibir_mensaje_externo(
        self, canal: str, id_cliente: str, texto: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registra un mensaje entrante desde WhatsApp, Instagram o Facebook."""
        canal_normalizado = canal if canal in self.canales_activos else "WhatsApp"
        registro = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "canal": canal_normalizado,
            "id_cliente": id_cliente,
            "texto": texto,
            "metadata": metadata or {},
        }
        self.bandeja_entrada.append(registro)
        return registro

    def emitir_respuesta_canal(
        self, canal: str, id_cliente: str, accion: str, contenido: Any
    ) -> Dict[str, Any]:
        """Registra una respuesta emitida hacia la red social del cliente."""
        registro = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "canal": canal,
            "id_cliente": id_cliente,
            "accion": accion,
            "contenido": contenido,
            "estado": "mensaje_enviado",
        }
        self.bandeja_salida.append(registro)
        return registro


@dataclass
class Informacion:
    """Representa el atributo `perfilesClientes : Informacion` del entorno."""

    perfiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def obtener_o_crear_perfil(
        self,
        id_cliente: str,
        nombre: str = "Cliente Potencial",
        canal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Obtiene el perfil del cliente o inicializa uno nuevo sin sobreescribir el canal."""
        if id_cliente not in self.perfiles:
            self.perfiles[id_cliente] = {
                "id_cliente": id_cliente,
                "nombre": nombre,
                "canal": canal or "WhatsApp",
                "historial_interacciones": [],
                "tono_actual": "neutral",
                "fidelidad": 1,
                "ultima_cotizacion": None,
                "ultima_complejidad": 3,
                "parametros_parciales": {},
                "esperando_parametros_cotizacion": False,
                "promocion_aplicada": None,
                "automatizacion_activa": True,
            }
        else:
            if nombre and nombre != "Cliente Potencial":
                self.perfiles[id_cliente]["nombre"] = nombre
            if canal is not None:
                self.perfiles[id_cliente]["canal"] = canal
        return self.perfiles[id_cliente]

    def actualizar_perfil(self, id_cliente: str, **campos: Any) -> Dict[str, Any]:
        """Actualiza campos puntuales en el perfil de un cliente."""
        perfil = self.obtener_o_crear_perfil(id_cliente)
        perfil.update(campos)
        return perfil


@dataclass
class BaseDatos:
    """
    Representa el atributo `tarifasYMargenes : BaseDatos` del entorno,
    con histórico de cotizaciones, ofertas y topes de descuento.
    """

    tarifas_historicas: pd.DataFrame
    historico_ofertas: pd.DataFrame
    margen_minimo_ganancia: float = 0.22
    margen_base_estimado: float = 0.34
    descuento_maximo_permitido: float = 15.0
    promociones_vigentes: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "codigo": "PROMO_CIERRE_10",
                "descripcion": "Incentivo de cierre rápido para clientes indecisos",
                "descuento_max": 12.0,
                "activa": True,
            },
            {
                "codigo": "PROMO_FIDELIDAD_15",
                "descripcion": "Beneficio tope de retención comercial",
                "descuento_max": 15.0,
                "activa": True,
            },
        ]
    )

    def consultar_referencias_servicio(self, tipo_servicio: str) -> pd.DataFrame:
        """Filtra cotizaciones históricas asociadas al servicio solicitado."""
        if "tipo_servicio" not in self.tarifas_historicas.columns:
            return self.tarifas_historicas
        filtrado = self.tarifas_historicas[
            self.tarifas_historicas["tipo_servicio"] == tipo_servicio
        ]
        return filtrado if not filtrado.empty else self.tarifas_historicas.head(10)


@dataclass
class Calendario:
    """Representa el atributo `agendaVirtual : Calendario` del entorno."""

    bloques: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "2026-11-18 10:00": {"estado": "LIBRE", "cliente": None},
            "2026-11-18 14:00": {"estado": "OCUPADO", "cliente": "Empresa Alfa SAS"},
            "2026-11-18 16:00": {"estado": "LIBRE", "cliente": None},
            "2026-11-19 09:00": {"estado": "LIBRE", "cliente": None},
            "2026-11-19 11:00": {"estado": "LIBRE", "cliente": None},
        }
    )

    def verificar_disponibilidad(self, fecha_hora: str) -> bool:
        """Retorna True si el bloque está libre o aún no figura ocupado."""
        bloque = self.bloques.get(fecha_hora)
        if bloque is None:
            return True
        return bloque["estado"] == "LIBRE"

    def obtener_alternativas(self, excluir_fecha: str, limite: int = 2) -> List[str]:
        """Retorna bloques horarios libres distintos a la fecha ocupada."""
        alternativas = [
            fh
            for fh, info in self.bloques.items()
            if info["estado"] == "LIBRE" and fh != excluir_fecha
        ]
        return alternativas[:limite]

    def cambiar_estado_bloque(
        self, fecha_hora: str, estado: str, cliente: Optional[str] = None
    ) -> Dict[str, Any]:
        """Actualiza un bloque en el calendario (LIBRE u OCUPADO)."""
        estado_norm = estado.upper()
        self.bloques[fecha_hora] = {"estado": estado_norm, "cliente": cliente}
        return {"fecha_hora": fecha_hora, "estado": estado_norm, "cliente": cliente}


class EntornoVirtualAtencion:
    """
    Entorno compartido MAS-ML (<<environment>> EntornoVirtualAtencion) donde
    habitan (<<inhabit>>) la organización SistemaMultiagenteComercial y los 5 agentes.
    Características del diseño conceptual: Parcialmente observable, Estocástico,
    Secuencial, Dinámico, Discreto y Multiagente Cooperativo.
    """

    def __init__(
        self,
        tarifas_historicas: pd.DataFrame,
        historico_ofertas: pd.DataFrame,
        logger_evento: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.plataformas: RedesSociales = RedesSociales()
        self.perfilesClientes: Informacion = Informacion()
        self.tarifasYMargenes: BaseDatos = BaseDatos(
            tarifas_historicas=tarifas_historicas,
            historico_ofertas=historico_ofertas,
        )
        self.agendaVirtual: Calendario = Calendario()

        self.habitantes: List[str] = []
        self.alertas_humanas: List[Dict[str, Any]] = []
        self.eventos_disparados: List[Dict[str, Any]] = []
        self._logger_evento = logger_evento

    def configurar_logger_evento(self, callback: Callable[[str, str], None]) -> None:
        """Conecta el monitor en tiempo real para trazar eventos del entorno."""
        self._logger_evento = callback

    def registrar_habitante(self, nombre_entidad: str) -> None:
        """Materializa la relación <<inhabit>> entre entorno y organización/agentes."""
        if nombre_entidad not in self.habitantes:
            self.habitantes.append(nombre_entidad)

    def _emitir_traza(self, origen: str, detalle: str) -> None:
        if self._logger_evento is not None:
            self._logger_evento(origen, detalle)

    # ------------------------------------------------------------------
    # EVENTOS DEL ENTORNO (no son acciones de agente)
    # ------------------------------------------------------------------
    def eventoLlegadaMensaje(
        self,
        tipo_evento: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evento del entorno: ingresa un mensaje de un cliente desde redes sociales
        y se prepara como percepción para AgenteVentas.
        """
        id_cliente = str(payload.get("id_cliente", "cliente_001"))
        canal = str(payload.get("canal", "WhatsApp"))
        texto = str(payload.get("texto", ""))
        nombre = str(payload.get("nombre", "Cliente Empresarial"))
        fidelidad = int(payload.get("fidelidad", 1))

        perfil = self.perfilesClientes.obtener_o_crear_perfil(
            id_cliente=id_cliente, nombre=nombre, canal=canal
        )
        perfil["fidelidad"] = fidelidad
        perfil["historial_interacciones"].append({"rol": "cliente", "texto": texto})

        entrada = self.plataformas.recibir_mensaje_externo(
            canal=canal,
            id_cliente=id_cliente,
            texto=texto,
            metadata={"tipo_evento": tipo_evento, **payload},
        )
        evento = {
            "evento": f"eventoLlegadaMensaje({tipo_evento})",
            "tipo_evento": tipo_evento,
            "entrada": entrada,
            "payload": payload,
        }
        self.eventos_disparados.append(evento)
        self._emitir_traza(
            "EntornoVirtualAtencion -> AgenteVentas",
            f"eventoLlegadaMensaje({tipo_evento}) | Canal={canal} | '{texto}'",
        )
        return evento

    def eventoCambioDisponibilidad(
        self,
        fecha_hora: str,
        estado: str,
        origen: str = "profesional_humano",
        cliente: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evento dinámico del entorno: el dueño o una sincronización externa modifica
        en tiempo real la disponibilidad en agendaVirtual mientras operan los agentes.
        """
        resultado = self.agendaVirtual.cambiar_estado_bloque(
            fecha_hora=fecha_hora, estado=estado, cliente=cliente
        )
        evento = {
            "evento": "eventoCambioDisponibilidad()",
            "origen": origen,
            **resultado,
        }
        self.eventos_disparados.append(evento)
        self._emitir_traza(
            "EntornoVirtualAtencion",
            f"eventoCambioDisponibilidad({fecha_hora}, estado={estado.upper()}, origen={origen})",
        )
        return evento

    # ------------------------------------------------------------------
    # INTERACCIÓN DE PERCEPCIÓN Y ACTUACIÓN PARA LOS AGENTES
    # ------------------------------------------------------------------
    def consultarTarifasHistoricas(self, parametros_servicio: Dict[str, Any]) -> Dict[str, Any]:
        """Retorna datos(tarifas_historicas) solicitados por AgenteCotizaciones."""
        tipo = str(parametros_servicio.get("tipo_servicio", "desarrollo_web"))
        referencias = self.tarifasYMargenes.consultar_referencias_servicio(tipo)
        self._emitir_traza(
            "AgenteCotizaciones <-> EntornoVirtualAtencion",
            f"consultarTarifasHistoricas(servicio={tipo}) -> datos(tarifas_historicas: {len(referencias)} registros)",
        )
        return {
            "registros": len(referencias),
            "tarifa_promedio_referencia": float(referencias["costo_total_cop"].mean()),
            "margen_estimado_base": self.tarifasYMargenes.margen_base_estimado,
            "dataframe": referencias,
        }

    def consultarHistoricoYMargenes(self, cotizacion_actual: float) -> Dict[str, Any]:
        """Retorna datos(margen_minimo, historico_ofertas) para AgentePromociones."""
        self._emitir_traza(
            "AgentePromociones <-> EntornoVirtualAtencion",
            (
                f"consultarHistoricoYMargenes(cotizacion={cotizacion_actual:,.0f}) -> "
                f"datos(margen_minimo={self.tarifasYMargenes.margen_minimo_ganancia:.0%}, "
                f"historico_ofertas={len(self.tarifasYMargenes.historico_ofertas)} registros)"
            ),
        )
        return {
            "margen_minimo": self.tarifasYMargenes.margen_minimo_ganancia,
            "margen_base": self.tarifasYMargenes.margen_base_estimado,
            "descuento_maximo_permitido": self.tarifasYMargenes.descuento_maximo_permitido,
            "promociones_vigentes": self.tarifasYMargenes.promociones_vigentes,
            "historico_ofertas": self.tarifasYMargenes.historico_ofertas,
        }

    def consultarCalendario(self, fecha_hora: str) -> Dict[str, Any]:
        """Retorna respuesta(estado_calendario) para AgenteAgendamientos."""
        disponible = self.agendaVirtual.verificar_disponibilidad(fecha_hora)
        alternativas = self.agendaVirtual.obtener_alternativas(excluir_fecha=fecha_hora)
        estado_str = "LIBRE" if disponible else "OCUPADO"
        self._emitir_traza(
            "AgenteAgendamientos <-> EntornoVirtualAtencion",
            f"consultarCalendario({fecha_hora}) -> respuesta(estado_calendario={estado_str})",
        )
        return {
            "fecha_hora": fecha_hora,
            "disponible": disponible,
            "estado": estado_str,
            "alternativas": alternativas,
        }

    def registrarCitaEnCalendario(
        self, fecha_hora: str, cliente: str, estado: str = "OCUPADO"
    ) -> Dict[str, Any]:
        """Registra de forma efectiva la reserva en agendaVirtual y retorna ok()."""
        registro = self.agendaVirtual.cambiar_estado_bloque(
            fecha_hora=fecha_hora, estado=estado, cliente=cliente
        )
        self._emitir_traza(
            "AgenteAgendamientos -> EntornoVirtualAtencion",
            f"administrarCitaYNotificar({fecha_hora}, {estado}) -> ok()",
        )
        return {"status": "ok", **registro}

    def registrarRespuestaEntorno(
        self, accion: str, canal: str, id_cliente: str, contenido: Any
    ) -> Dict[str, Any]:
        """Registra las acciones comunicativas de AgenteVentas hacia el entorno."""
        perfil = self.perfilesClientes.obtener_o_crear_perfil(id_cliente, canal=canal)
        perfil["historial_interacciones"].append({"rol": "agente_ventas", "accion": accion, "contenido": contenido})
        salida = self.plataformas.emitir_respuesta_canal(
            canal=canal, id_cliente=id_cliente, accion=accion, contenido=contenido
        )
        self._emitir_traza(
            "AgenteVentas -> EntornoVirtualAtencion",
            f"{accion} -> confirmacion(mensaje_enviado)",
        )
        return salida

    def notificarAlertaHumana(
        self,
        historial_charla: List[Dict[str, Any]],
        motivo_critico: str,
        id_cliente: str = "cliente_001",
    ) -> Dict[str, Any]:
        """Pausa la automatizacion del cliente y alerta al profesional humano."""
        perfil = self.perfilesClientes.actualizar_perfil(
            id_cliente, automatizacion_activa=False
        )
        alerta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "id_cliente": id_cliente,
            "nombre": perfil.get("nombre", "Cliente"),
            "canal": perfil.get("canal", "WhatsApp"),
            "motivo_critico": motivo_critico,
            "historial_charla": historial_charla,
            "estado": "alerta_recibida_por_profesional",
            "resuelta": False,
        }
        self.alertas_humanas.append(alerta)
        self._emitir_traza(
            "AgenteCoordinador -> EntornoVirtualAtencion",
            f"pausarYAlertarHumano(motivo='{motivo_critico}') -> ok(alerta_recibida_por_profesional)",
        )
        return alerta

