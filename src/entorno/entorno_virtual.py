"""
Implementacion del entorno compartido <<environment>> EntornoVirtualAtencion
segun el Diagrama de Clases MAS-ML y la Tabla de Analisis del Entorno,
desacoplado mediante RepositorioEntorno (Supabase en la nube o SQLite local).

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.ml.datasets import CATALOGO_SERVICIOS
from src.persistencia.repositorio import (
    RepositorioEntorno,
    RepositorioSQLite,
)


@dataclass
class RedesSociales:
    """Representa el atributo `plataformas : RedesSociales` del entorno."""

    canales_activos: List[str] = field(
        default_factory=lambda: ["WhatsApp", "Instagram", "Facebook", "Webchat", "Botpress"]
    )
    portafolio_servicios: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: dict(CATALOGO_SERVICIOS)
    )
    bandeja_entrada: List[Dict[str, Any]] = field(default_factory=list)
    bandeja_salida: List[Dict[str, Any]] = field(default_factory=list)

    def recibir_mensaje_externo(
        self, canal: str, id_cliente: str, texto: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registra un mensaje entrante desde cualquier canal conectado via Botpress."""
        canal_normalizado = canal.strip() or "WhatsApp"
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
    """Representa el atributo `perfilesClientes : Informacion` sincronizado con BD."""

    perfiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _on_save: Optional[Callable[[Dict[str, Any]], None]] = field(default=None, repr=False)

    def obtener_o_crear_perfil(
        self,
        id_cliente: str,
        nombre: str = "Cliente Potencial",
        canal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Obtiene el perfil del cliente o inicializa uno nuevo persistiendo en BD."""
        cambio = False
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
            cambio = True
        else:
            if nombre and nombre != "Cliente Potencial" and self.perfiles[id_cliente].get("nombre") != nombre:
                self.perfiles[id_cliente]["nombre"] = nombre
                cambio = True
            if canal is not None and self.perfiles[id_cliente].get("canal") != canal:
                self.perfiles[id_cliente]["canal"] = canal
                cambio = True
        if cambio and self._on_save is not None:
            self._on_save(self.perfiles[id_cliente])
        return self.perfiles[id_cliente]

    def actualizar_perfil(self, id_cliente: str, **campos: Any) -> Dict[str, Any]:
        """Actualiza campos puntuales en el perfil del cliente y los guarda en BD."""
        perfil = self.obtener_o_crear_perfil(id_cliente)
        perfil.update(campos)
        if self._on_save is not None:
            self._on_save(perfil)
        return perfil


@dataclass
class BaseDatos:
    """Representa el atributo `tarifasYMargenes : BaseDatos` del entorno."""

    tarifas_historicas: pd.DataFrame
    historico_ofertas: pd.DataFrame
    margen_base_estimado: float = 0.35
    margen_minimo_ganancia: float = 0.22
    descuento_maximo_permitido: float = 15.0
    promociones_vigentes: List[Dict[str, Any]] = field(default_factory=list)

    def consultar_referencias_servicio(self, tipo_servicio: str) -> pd.DataFrame:
        """Filtra cotizaciones historicas asociadas al servicio solicitado."""
        if "tipo_servicio" not in self.tarifas_historicas.columns:
            return self.tarifas_historicas
        filtrado = self.tarifas_historicas[
            self.tarifas_historicas["tipo_servicio"] == tipo_servicio
        ]
        return filtrado if not filtrado.empty else self.tarifas_historicas.head(10)


@dataclass
class Calendario:
    """Representa el atributo `agendaVirtual : Calendario` sincronizado con BD."""

    bloques: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _on_save: Optional[Callable[[str, str, Optional[str]], None]] = field(
        default=None, repr=False
    )

    def verificar_disponibilidad(self, fecha_hora: str) -> bool:
        """Retorna True si el bloque esta libre o aun no figura ocupado."""
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
        """Actualiza la disponibilidad del bloque en memoria y en BD."""
        estado_norm = estado.upper()
        self.bloques[fecha_hora] = {"estado": estado_norm, "cliente": cliente}
        if self._on_save is not None:
            self._on_save(fecha_hora, estado_norm, cliente)
        return {"fecha_hora": fecha_hora, "estado": estado_norm, "cliente": cliente}


class EntornoVirtualAtencion:
    """
    <<environment>> EntornoVirtualAtencion:
    Espacio compartido donde habitan la organizacion y los 5 agentes SPADE.
    Toda la configuracion comercial, catalogo, historico ML, agenda y estado
    Human-in-the-Loop se carga y persiste mediante `RepositorioEntorno`.
    """

    def __init__(
        self,
        tarifas_historicas: Optional[pd.DataFrame] = None,
        historico_ofertas: Optional[pd.DataFrame] = None,
        logger_evento: Optional[Callable[[str, str], None]] = None,
        repositorio: Optional[RepositorioEntorno] = None,
    ) -> None:
        self.repositorio: RepositorioEntorno = repositorio or RepositorioSQLite(":memory:")
        self.config_negocio: Dict[str, Any] = self.repositorio.obtener_configuracion_negocio()
        catalogo = self.repositorio.obtener_catalogo_servicios() or dict(CATALOGO_SERVICIOS)

        df_tarifas = (
            tarifas_historicas
            if tarifas_historicas is not None
            else self.repositorio.obtener_tarifas_historicas()
        )
        df_ofertas = (
            historico_ofertas
            if historico_ofertas is not None
            else self.repositorio.obtener_historico_ofertas()
        )

        self.plataformas: RedesSociales = RedesSociales(portafolio_servicios=catalogo)
        self.perfilesClientes: Informacion = Informacion(
            perfiles=self.repositorio.listar_perfiles_clientes(),
            _on_save=self.repositorio.guardar_perfil_cliente,
        )
        self.tarifasYMargenes: BaseDatos = BaseDatos(
            tarifas_historicas=df_tarifas,
            historico_ofertas=df_ofertas,
            margen_base_estimado=float(self.config_negocio.get("margen_base", 0.35)),
            margen_minimo_ganancia=float(self.config_negocio.get("margen_minimo", 0.22)),
            descuento_maximo_permitido=float(
                self.config_negocio.get("descuento_maximo", 15.0)
            ),
            promociones_vigentes=self.repositorio.obtener_promociones_vigentes(),
        )
        self.agendaVirtual: Calendario = Calendario(
            bloques=self.repositorio.obtener_bloques_agenda(),
            _on_save=self.repositorio.guardar_bloque_agenda,
        )

        self.habitantes: List[str] = []
        self.alertas_humanas: List[Dict[str, Any]] = (
            self.repositorio.listar_alertas_humanas()
        )
        self.eventos_disparados: List[Dict[str, Any]] = []
        self._logger_evento = logger_evento

    def configurar_logger_evento(self, callback: Callable[[str, str], None]) -> None:
        """Conecta el monitor en tiempo real para trazar eventos del entorno."""
        self._logger_evento = callback

    def registrar_habitante(self, nombre_entidad: str) -> None:
        """Materializa la relacion <<inhabit>> entre entorno y organizacion/agentes."""
        if nombre_entidad not in self.habitantes:
            self.habitantes.append(nombre_entidad)

    def _emitir_traza(self, origen: str, detalle: str) -> None:
        if self._logger_evento is not None:
            self._logger_evento(origen, detalle)

    # ------------------------------------------------------------------
    # EVENTOS DEL ENTORNO (NO ACCIONES DE AGENTE)
    # ------------------------------------------------------------------
    def eventoLlegadaMensaje(
        self,
        tipo_evento: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evento del entorno: ingresa un mensaje de un cliente desde redes sociales
        y se prepara como percepcion para AgenteVentas.
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
        self.repositorio.guardar_perfil_cliente(perfil)

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
        Evento dinamico del entorno: modifica en tiempo real la disponibilidad
        en agendaVirtual y persiste en base de datos.
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
    # INTERACCION DE PERCEPCION Y ACTUACION PARA LOS AGENTES
    # ------------------------------------------------------------------
    def consultarTarifasHistoricas(self, parametros_servicio: Dict[str, Any]) -> Dict[str, Any]:
        """Retorna datos(tarifas_historicas) solicitados por AgenteCotizaciones."""
        clave_default = next(iter(self.plataformas.portafolio_servicios.keys()), "general")
        tipo = str(parametros_servicio.get("tipo_servicio", clave_default))
        referencias = self.tarifasYMargenes.consultar_referencias_servicio(tipo)
        col_total = "costo_total" if "costo_total" in referencias.columns else "costo_total_cop"
        promedio = float(referencias[col_total].mean()) if not referencias.empty else 0.0
        self._emitir_traza(
            "AgenteCotizaciones <-> EntornoVirtualAtencion",
            f"consultarTarifasHistoricas(servicio={tipo}) -> datos(tarifas_historicas: {len(referencias)} registros)",
        )
        return {
            "registros": len(referencias),
            "tarifa_promedio_referencia": promedio,
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
        """Registra de forma efectiva la reserva en agendaVirtual y en BD."""
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
        """Registra las acciones comunicativas de AgenteVentas y las persiste."""
        perfil = self.perfilesClientes.obtener_o_crear_perfil(id_cliente, canal=canal)
        perfil["historial_interacciones"].append(
            {"rol": "agente_ventas", "accion": accion, "contenido": contenido}
        )
        self.repositorio.guardar_perfil_cliente(perfil)
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
        """Pausa la automatizacion del cliente en BD y registra la alerta humana."""
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
        self.repositorio.registrar_alerta_humana(alerta)
        self._emitir_traza(
            "AgenteCoordinador -> EntornoVirtualAtencion",
            f"pausarYAlertarHumano(motivo='{motivo_critico}') -> ok(alerta_recibida_por_profesional)",
        )
        return alerta
