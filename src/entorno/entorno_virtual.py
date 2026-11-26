"""
Implementacion del entorno compartido <<environment>> EntornoVirtualAtencion
segun el Diagrama de Clases MAS-ML y la Tabla de Analisis del Entorno,
desacoplado mediante RepositorioEntorno (Supabase en la nube o SQLite local).
"""
from __future__ import annotations

import sqlite3
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from src.ml.datasets import CATALOGO_SERVICIOS
from src.persistencia.google_calendar import ClienteGoogleCalendar, ErrorGoogleCalendar
from src.reglas_comerciales import validar_datos_cliente_reserva
from src.persistencia.repositorio import (
    ConflictoAgendamientoError,
    RepositorioEntorno,
    RepositorioSQLite,
)

_CLAVE_ESTADO_COTIZACION = "_estado_cotizacion"
_CLAVE_CAMPOS_PENDIENTES_COTIZACION = "_campos_pendientes_cotizacion"
_CLAVE_PREPERFILADO_COTIZACION = "_preperfilado_sin_escalar"
_CLAVE_CONTEXTOS_CONVERSACION = "_contextos_conversacion"


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

    def _obtener_contexto_conversacion(
        self, id_cliente: str, conversation_id: Optional[str]
    ) -> Dict[str, Any]:
        if not conversation_id:
            perfil = self.perfiles.get(id_cliente, {})
            parametros = perfil.get("parametros_parciales", {})
            estado_reserva = (
                parametros.get("_estado_reserva_legacy", {})
                if isinstance(parametros, dict)
                else {}
            )
            return {"estado_reserva": dict(estado_reserva)} if isinstance(
                estado_reserva, dict
            ) else {}
        perfil = self.perfiles.get(id_cliente, {})
        parametros = perfil.get("parametros_parciales", {})
        contextos = (
            parametros.get(_CLAVE_CONTEXTOS_CONVERSACION, {})
            if isinstance(parametros, dict)
            else {}
        )
        contexto = contextos.get(str(conversation_id), {}) if isinstance(contextos, dict) else {}
        return dict(contexto) if isinstance(contexto, dict) else {}

    def _guardar_contexto_conversacion(
        self, id_cliente: str, conversation_id: str, contexto: Dict[str, Any]
    ) -> None:
        perfil = self.obtener_o_crear_perfil(id_cliente)
        parametros = perfil.get("parametros_parciales", {})
        parametros_actuales = dict(parametros) if isinstance(parametros, dict) else {}
        contextos = parametros_actuales.get(_CLAVE_CONTEXTOS_CONVERSACION, {})
        contextos_actuales = dict(contextos) if isinstance(contextos, dict) else {}
        contextos_actuales[conversation_id] = contexto
        parametros_actuales[_CLAVE_CONTEXTOS_CONVERSACION] = contextos_actuales
        esperando_cotizacion = any(
            isinstance(item, dict)
            and item.get("esperando_parametros_cotizacion", False)
            for item in contextos_actuales.values()
        )
        self.actualizar_perfil(
            id_cliente,
            parametros_parciales=parametros_actuales,
            esperando_parametros_cotizacion=esperando_cotizacion,
        )

    def obtener_estado_conversacion(
        self, id_cliente: str, conversation_id: Optional[str]
    ) -> Dict[str, Any]:
        """Obtiene el contexto comercial aislado de una conversación Botpress."""
        return self._obtener_contexto_conversacion(id_cliente, conversation_id)

    def actualizar_estado_conversacion(
        self,
        id_cliente: str,
        conversation_id: Optional[str],
        cambios: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persiste estado comercial y datos de reserva bajo la conversación."""
        if not conversation_id:
            perfil = self.obtener_o_crear_perfil(id_cliente)
            parametros = perfil.get("parametros_parciales", {})
            parametros_actuales = (
                dict(parametros) if isinstance(parametros, dict) else {}
            )
            estado = parametros_actuales.get("_estado_reserva_legacy", {})
            estado_actual = dict(estado) if isinstance(estado, dict) else {}
            if isinstance(cambios.get("estado_reserva"), dict):
                estado_actual = dict(cambios["estado_reserva"])
            else:
                estado_actual.update(cambios)
            parametros_actuales["_estado_reserva_legacy"] = estado_actual
            self.actualizar_perfil(
                id_cliente, parametros_parciales=parametros_actuales
            )
            return {"estado_reserva": estado_actual}
        contexto = self._obtener_contexto_conversacion(id_cliente, conversation_id)
        contexto.update(cambios)
        self._guardar_contexto_conversacion(id_cliente, conversation_id, contexto)
        return contexto

    def obtener_parametros_cotizacion(
        self, id_cliente: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retorna los parametros de cotizacion sin el estado interno de aprobacion."""
        perfil = self.perfiles.get(id_cliente, {})
        parametros = perfil.get("parametros_parciales", {})
        if not isinstance(parametros, dict):
            return {}
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = contexto.get("parametros_cotizacion", {})
            if not isinstance(parametros, dict):
                return {}
        return {
            clave: valor
            for clave, valor in parametros.items()
            if clave
            not in {
                _CLAVE_ESTADO_COTIZACION,
                _CLAVE_CAMPOS_PENDIENTES_COTIZACION,
                _CLAVE_PREPERFILADO_COTIZACION,
                _CLAVE_CONTEXTOS_CONVERSACION,
            }
        }

    def es_preperfilado_cotizacion(
        self, id_cliente: str, conversation_id: Optional[str] = None
    ) -> bool:
        """Indica si los parametros pendientes vienen de perfilamiento previo sin FAILURE."""
        perfil = self.perfiles.get(id_cliente, {})
        parametros = perfil.get("parametros_parciales", {})
        if not isinstance(parametros, dict):
            return False
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = contexto.get("parametros_cotizacion", {})
            if not isinstance(parametros, dict):
                return False
        return bool(parametros.get(_CLAVE_PREPERFILADO_COTIZACION, False))

    def obtener_campos_cotizacion_pendientes(
        self, id_cliente: str, conversation_id: Optional[str] = None
    ) -> List[str]:
        """Obtiene los campos exactos solicitados por la ultima validacion."""
        perfil = self.perfiles.get(id_cliente, {})
        parametros = perfil.get("parametros_parciales", {})
        if not isinstance(parametros, dict):
            return []
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = contexto.get("parametros_cotizacion", {})
            if not isinstance(parametros, dict):
                return []
        campos = parametros.get(_CLAVE_CAMPOS_PENDIENTES_COTIZACION, [])
        return [str(campo) for campo in campos] if isinstance(campos, list) else []

    def actualizar_parametros_cotizacion(
        self,
        id_cliente: str,
        cambios: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Actualiza parámetros comerciales sin alterar aprobación ni campos pendientes."""
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = contexto.get("parametros_cotizacion", {})
            parametros_actuales = (
                dict(parametros) if isinstance(parametros, dict) else {}
            )
            parametros_actuales.update(cambios)
            contexto["parametros_cotizacion"] = parametros_actuales
            self._guardar_contexto_conversacion(
                id_cliente, conversation_id, contexto
            )
            return self.perfiles[id_cliente]
        perfil = self.obtener_o_crear_perfil(id_cliente)
        parametros = perfil.get("parametros_parciales", {})
        parametros_actuales = dict(parametros) if isinstance(parametros, dict) else {}
        parametros_actuales.update(cambios)
        return self.actualizar_perfil(
            id_cliente, parametros_parciales=parametros_actuales
        )

    def registrar_parametros_cotizacion_pendientes(
        self,
        id_cliente: str,
        parametros: Dict[str, Any],
        campos: List[str],
        conversation_id: Optional[str] = None,
        preperfilado: bool = False,
    ) -> Dict[str, Any]:
        """Persiste parametros y campos faltantes para continuar en otro turno."""
        parametros_persistidos = dict(parametros)
        parametros_persistidos[_CLAVE_CAMPOS_PENDIENTES_COTIZACION] = list(campos)
        parametros_persistidos[_CLAVE_PREPERFILADO_COTIZACION] = bool(preperfilado)
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            contexto["parametros_cotizacion"] = parametros_persistidos
            contexto["esperando_parametros_cotizacion"] = True
            self._guardar_contexto_conversacion(
                id_cliente, conversation_id, contexto
            )
            return self.perfiles[id_cliente]
        return self.actualizar_perfil(
            id_cliente,
            parametros_parciales=parametros_persistidos,
            esperando_parametros_cotizacion=True,
        )

    def establecer_fecha_agendamiento_pendiente(
        self,
        id_cliente: str,
        fecha_hora: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> None:
        """Guarda o limpia la solicitud de horario sin alterar otros parametros."""
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = contexto.get("parametros_cotizacion", {})
            parametros_actuales = (
                dict(parametros) if isinstance(parametros, dict) else {}
            )
            if fecha_hora:
                parametros_actuales["fecha_agendamiento_pendiente"] = fecha_hora
                parametros_actuales.pop("fecha_agendamiento_fecha", None)
                parametros_actuales.pop("fecha_hora_falta", None)
            else:
                for clave in (
                    "fecha_agendamiento_pendiente",
                    "fecha_agendamiento_fecha",
                    "fecha_hora",
                    "fecha_hora_falta",
                ):
                    parametros_actuales.pop(clave, None)
            contexto["parametros_cotizacion"] = parametros_actuales
            self._guardar_contexto_conversacion(
                id_cliente, conversation_id, contexto
            )
            return
        perfil = self.obtener_o_crear_perfil(id_cliente)
        parametros = perfil.get("parametros_parciales", {})
        parametros_actuales = dict(parametros) if isinstance(parametros, dict) else {}
        if fecha_hora:
            parametros_actuales["fecha_agendamiento_pendiente"] = fecha_hora
        else:
            parametros_actuales.pop("fecha_agendamiento_pendiente", None)
        self.actualizar_perfil(id_cliente, parametros_parciales=parametros_actuales)

    def tiene_cotizacion(
        self, id_cliente: str, conversation_id: Optional[str] = None
    ) -> bool:
        perfil = self.perfiles.get(id_cliente)
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            return bool(
                contexto.get("ultima_cotizacion") is not None
                and not contexto.get("esperando_parametros_cotizacion", False)
            )
        return bool(
            perfil
            and perfil.get("ultima_cotizacion") is not None
            and not perfil.get("esperando_parametros_cotizacion", False)
        )

    def cotizacion_aprobada(
        self, id_cliente: str, conversation_id: Optional[str] = None
    ) -> bool:
        perfil = self.perfiles.get(id_cliente)
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = contexto.get("parametros_cotizacion", {})
            estado = (
                parametros.get(_CLAVE_ESTADO_COTIZACION)
                if isinstance(parametros, dict)
                else None
            )
            if (
                contexto.get("ultima_cotizacion") is None
                or contexto.get("esperando_parametros_cotizacion", False)
                or not isinstance(estado, dict)
                or estado.get("aprobada") is not True
            ):
                return False
            try:
                return float(estado.get("valor_total")) == float(
                    contexto["ultima_cotizacion"]
                )
            except (TypeError, ValueError):
                return False
        if (
            not perfil
            or perfil.get("ultima_cotizacion") is None
            or perfil.get("esperando_parametros_cotizacion", False)
        ):
            return False
        parametros = perfil.get("parametros_parciales", {})
        if not isinstance(parametros, dict):
            return False
        estado = parametros.get(_CLAVE_ESTADO_COTIZACION)
        if not isinstance(estado, dict) or estado.get("aprobada") is not True:
            return False
        try:
            return float(estado.get("valor_total")) == float(
                perfil["ultima_cotizacion"]
            )
        except (TypeError, ValueError):
            return False

    def registrar_cotizacion_calculada(
        self,
        id_cliente: str,
        valor_total: float,
        complejidad: int,
        parametros_cotizacion: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persiste una cotizacion nueva como pendiente de aprobacion del cliente."""
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            parametros = dict(contexto.get("parametros_cotizacion", {}))
            parametros.update(parametros_cotizacion or {})
            parametros[_CLAVE_ESTADO_COTIZACION] = {
                "valor_total": float(valor_total),
                "aprobada": False,
            }
            contexto.update(
                {
                    "parametros_cotizacion": parametros,
                    "ultima_cotizacion": float(valor_total),
                    "ultima_complejidad": int(complejidad),
                    "esperando_parametros_cotizacion": False,
                }
            )
            self._guardar_contexto_conversacion(
                id_cliente, conversation_id, contexto
            )
            return self.actualizar_perfil(
                id_cliente,
                ultima_cotizacion=float(valor_total),
                ultima_complejidad=int(complejidad),
            )
        parametros = self.obtener_parametros_cotizacion(id_cliente)
        parametros.update(parametros_cotizacion or {})
        parametros[_CLAVE_ESTADO_COTIZACION] = {
            "valor_total": float(valor_total),
            "aprobada": False,
        }
        return self.actualizar_perfil(
            id_cliente,
            ultima_cotizacion=float(valor_total),
            ultima_complejidad=int(complejidad),
            parametros_parciales=parametros,
            esperando_parametros_cotizacion=False,
        )

    def aprobar_ultima_cotizacion(
        self, id_cliente: str, conversation_id: Optional[str] = None
    ) -> bool:
        """Marca aprobada la ultima cotizacion existente para un cliente."""
        perfil = self.perfiles.get(id_cliente)
        if conversation_id:
            contexto = self._obtener_contexto_conversacion(
                id_cliente, conversation_id
            )
            if (
                contexto.get("ultima_cotizacion") is None
                or contexto.get("esperando_parametros_cotizacion", False)
            ):
                return False
            parametros = contexto.get("parametros_cotizacion", {})
            parametros_actuales = (
                dict(parametros) if isinstance(parametros, dict) else {}
            )
            estado = parametros_actuales.get(_CLAVE_ESTADO_COTIZACION)
            valor_total = float(contexto["ultima_cotizacion"])
            if not isinstance(estado, dict):
                estado = {"valor_total": valor_total, "aprobada": False}
            parametros_actuales[_CLAVE_ESTADO_COTIZACION] = {
                **estado,
                "valor_total": valor_total,
                "aprobada": True,
            }
            contexto["parametros_cotizacion"] = parametros_actuales
            self._guardar_contexto_conversacion(
                id_cliente, conversation_id, contexto
            )
            return self.cotizacion_aprobada(id_cliente, conversation_id)
        if (
            not perfil
            or perfil.get("ultima_cotizacion") is None
            or perfil.get("esperando_parametros_cotizacion", False)
        ):
            return False
        try:
            valor_total = float(perfil["ultima_cotizacion"])
        except (TypeError, ValueError):
            return False

        parametros = perfil.get("parametros_parciales", {})
        parametros_actuales = dict(parametros) if isinstance(parametros, dict) else {}
        estado = parametros_actuales.get(_CLAVE_ESTADO_COTIZACION)
        if not isinstance(estado, dict):
            estado = {"valor_total": valor_total, "aprobada": False}
        else:
            try:
                if float(estado.get("valor_total")) != valor_total:
                    estado = {"valor_total": valor_total, "aprobada": False}
            except (TypeError, ValueError):
                estado = {"valor_total": valor_total, "aprobada": False}

        parametros_actuales[_CLAVE_ESTADO_COTIZACION] = {
            **estado,
            "valor_total": valor_total,
            "aprobada": True,
        }
        self.actualizar_perfil(
            id_cliente, parametros_parciales=parametros_actuales
        )
        return self.cotizacion_aprobada(id_cliente)


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


def _agenda_local_respaldo() -> Dict[str, Dict[str, Any]]:
    return {
        "2026-11-18 10:00": {"estado": "LIBRE", "cliente": None},
        "2026-11-18 14:00": {"estado": "OCUPADO", "cliente": "Empresa Alfa SAS"},
        "2026-11-18 16:00": {"estado": "LIBRE", "cliente": None},
        "2026-11-19 09:00": {"estado": "LIBRE", "cliente": None},
        "2026-11-19 11:00": {"estado": "LIBRE", "cliente": None},
    }


def _intervalo_agendamiento(
    fecha_hora: str, duracion_minutos: int = 60
) -> tuple[datetime, datetime]:
    try:
        inicio = datetime.fromisoformat(fecha_hora.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            "La fecha de la cita debe tener el formato YYYY-MM-DD HH:MM."
        ) from None
    zona = ZoneInfo("America/Bogota")
    if inicio.tzinfo is None:
        inicio = inicio.replace(tzinfo=zona)
    else:
        inicio = inicio.astimezone(zona)
    fin = inicio + timedelta(minutes=duracion_minutos)
    return inicio, fin


def _formato_fecha_hora_calendario(inicio: datetime) -> str:
    if inicio.second or inicio.microsecond:
        return inicio.isoformat(timespec="auto")
    return inicio.strftime("%Y-%m-%d %H:%M")


@dataclass
class Calendario:
    """
    Representa el atributo `agendaVirtual : Calendario`.
    Opera directamente contra Google Calendar API v3 cuando esta configurado,
    o con bloques locales en memoria para pruebas offline.
    """

    bloques: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    google_calendar: Optional[ClienteGoogleCalendar] = field(default=None, repr=False)
    repositorio: Optional[RepositorioEntorno] = field(default=None, repr=False)
    duracion_minutos: int = 60

    @property
    def modo(self) -> str:
        if self.google_calendar is not None:
            return "Google Calendar API v3"
        if self.repositorio is not None and "Supabase" in self.repositorio.modo_bd:
            return "Supabase (solo)"
        return "Local (Respaldo)"

    def verificar_disponibilidad(self, fecha_hora: str) -> bool:
        """Verifica cruces persistidos y, si está configurado, la disponibilidad real."""
        inicio, fin = _intervalo_agendamiento(fecha_hora, self.duracion_minutos)
        fecha_hora_calendario = _formato_fecha_hora_calendario(inicio)
        inicio_iso, fin_iso = inicio.isoformat(), fin.isoformat()
        ocupado_en_sistema = bool(
            self.repositorio
            and self.repositorio.listar_agendamientos_activos(
                inicio_iso, fin_iso
            )
        )
        if self.google_calendar is not None:
            try:
                libre_en_google = self.google_calendar.verificar_disponibilidad(
                    fecha_hora_calendario
                )
            except ErrorGoogleCalendar:
                if ocupado_en_sistema:
                    return False
                raise
            return not ocupado_en_sistema and libre_en_google
        if ocupado_en_sistema:
            return False
        bloque = self.bloques.get(fecha_hora_calendario)
        if bloque is None:
            return True
        return bloque["estado"] == "LIBRE"

    def obtener_alternativas(
        self,
        excluir_fecha: str,
        limite: int = 2,
        excluir_horarios: Optional[List[str]] = None,
    ) -> List[str]:
        """Propone bloques que no se crucen con citas persistidas ni con Google."""
        if limite <= 0:
            return []
        inicio_ref, _ = _intervalo_agendamiento(
            excluir_fecha, self.duracion_minutos
        )
        excluir_fecha_calendario = _formato_fecha_hora_calendario(inicio_ref)
        excluidos = set()
        for horario in excluir_horarios or []:
            try:
                inicio_excluido, _ = _intervalo_agendamiento(
                    str(horario), self.duracion_minutos
                )
            except ValueError:
                continue
            excluidos.add(_formato_fecha_hora_calendario(inicio_excluido))
        if self.google_calendar is not None:
            limite_candidatas = max(
                limite,
                (
                    (self.google_calendar.hora_fin_laboral
                    - self.google_calendar.hora_inicio_laboral)
                    * 60
                    // self.duracion_minutos
                ),
            )
            candidatas = self.google_calendar.obtener_alternativas(
                excluir_fecha=excluir_fecha_calendario,
                limite=limite_candidatas,
            )
        else:
            jornada_ini = inicio_ref.replace(hour=8, minute=0, second=0, microsecond=0)
            jornada_fin = inicio_ref.replace(hour=18, minute=0, second=0, microsecond=0)
            paso = timedelta(minutes=self.duracion_minutos)
            candidatas = []
            cursor = jornada_ini
            while cursor + paso <= jornada_fin:
                etiqueta = cursor.strftime("%Y-%m-%d %H:%M")
                if etiqueta != excluir_fecha_calendario:
                    candidatas.append(etiqueta)
                cursor += paso

        alternativas: List[str] = []
        _, fin_excluido = _intervalo_agendamiento(
            excluir_fecha_calendario, self.duracion_minutos
        )
        for candidata in candidatas:
            if candidata in excluidos:
                continue
            inicio_candidata, fin_candidata = _intervalo_agendamiento(
                candidata, self.duracion_minutos
            )
            if inicio_candidata < fin_excluido and fin_candidata > inicio_ref:
                continue
            if self.verificar_disponibilidad(candidata):
                alternativas.append(candidata)
                if len(alternativas) >= limite:
                    break
        return alternativas

    def cambiar_estado_bloque(
        self, fecha_hora: str, estado: str, cliente: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registra la cita en Google Calendar (si esta activo) y actualiza vista."""
        estado_norm = estado.upper()
        extra: Dict[str, Any] = {}
        if self.google_calendar is not None and estado_norm == "OCUPADO":
            extra = self.google_calendar.crear_evento(
                fecha_hora=fecha_hora, cliente=cliente or "Cliente Comercial"
            )
        self.bloques[fecha_hora] = {"estado": estado_norm, "cliente": cliente}
        return {"fecha_hora": fecha_hora, "estado": estado_norm, "cliente": cliente, **extra}

    def obtener_bloques_vista(self) -> Dict[str, Dict[str, Any]]:
        """Retorna proximos eventos reales de Google Calendar o bloques locales."""
        if self.google_calendar is not None:
            try:
                return self.google_calendar.listar_proximos_eventos()
            except Exception:
                return self.bloques
        return self.bloques


class EntornoVirtualAtencion:
    """
    <<environment>> EntornoVirtualAtencion:
    Espacio compartido donde habitan la organizacion y los 5 agentes SPADE.
    Toda la configuracion comercial, catalogo, historico ML, agenda y estado
    Human-in-the-Loop se carga y persiste mediante `RepositorioEntorno` y
    `ClienteGoogleCalendar`.
    """

    def __init__(
        self,
        tarifas_historicas: Optional[pd.DataFrame] = None,
        historico_ofertas: Optional[pd.DataFrame] = None,
        logger_evento: Optional[Callable[[str, str], None]] = None,
        repositorio: Optional[RepositorioEntorno] = None,
        google_calendar: Optional[ClienteGoogleCalendar] = None,
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
            bloques=(
                _agenda_local_respaldo()
                if google_calendar is None and "Supabase" not in self.repositorio.modo_bd
                else {}
            ),
            google_calendar=google_calendar,
            repositorio=self.repositorio,
            duracion_minutos=(
                google_calendar.duracion_minutos if google_calendar is not None else 60
            ),
        )

        self.habitantes: List[str] = []
        self.alertas_humanas: List[Dict[str, Any]] = (
            self.repositorio.listar_alertas_humanas()
        )
        self.eventos_disparados: List[Dict[str, Any]] = []
        self._logger_evento = logger_evento

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
        texto_historial = re.sub(
            r"(?<!\d)\+?\d[\d\s().-]{5,18}\d(?!\d)",
            "[dato numérico oculto]",
            texto,
        )
        perfil["historial_interacciones"].append(
            {"rol": "cliente", "texto": texto_historial}
        )
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
            f"eventoLlegadaMensaje({tipo_evento}) | Canal={canal} | mensaje_recibido",
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
        en agendaVirtual (y sincroniza en Google Calendar cuando esta activo).
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

    def consultarCalendario(
        self,
        fecha_hora: str,
        alternativas_excluidas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retorna respuesta(estado_calendario) para AgenteAgendamientos."""
        disponible = self.agendaVirtual.verificar_disponibilidad(fecha_hora)
        alternativas: List[str] = []
        if not disponible:
            try:
                alternativas = self.agendaVirtual.obtener_alternativas(
                    excluir_fecha=fecha_hora,
                    excluir_horarios=alternativas_excluidas,
                )
            except ErrorGoogleCalendar as exc:
                self._emitir_traza(
                    "EntornoVirtualAtencion",
                    "No se pudieron consultar horarios alternativos en Google Calendar "
                    f"({type(exc).__name__}).",
                )
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

    def validarDatosAgendamiento(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Valida y normaliza los datos mínimos antes de consultar o persistir."""
        fecha_hora = str(datos.get("fecha_hora", "")).strip()
        inicio, fin = _intervalo_agendamiento(
            fecha_hora, self.agendaVirtual.duracion_minutos
        )
        id_cliente = str(datos.get("id_cliente", "")).strip()
        nombre = str(datos.get("nombre_cliente", "")).strip()
        canal = str(datos.get("canal", "")).strip()
        if not id_cliente or not nombre or not canal:
            raise ValueError("La cita requiere id_cliente, nombre_cliente y canal.")
        servicio = str(datos.get("servicio", "")).strip() or None
        metadata = datos.get("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        datos_cliente = metadata.get("datos_cliente", {})
        if not isinstance(datos_cliente, dict):
            raise ValueError("Faltan los datos completos del cliente para la reserva.")
        datos_cliente_normalizados = validar_datos_cliente_reserva(
            datos_cliente.get("nombre_cliente", nombre),
            datos_cliente.get("cedula"),
            datos_cliente.get("telefono"),
        )
        if datos_cliente_normalizados["nombre_cliente"] != nombre:
            raise ValueError(
                "El nombre de la reserva debe coincidir con el dato recopilado."
            )
        metadata["datos_cliente"] = datos_cliente_normalizados
        return {
            "id_cliente": id_cliente,
            "nombre_cliente": datos_cliente_normalizados["nombre_cliente"],
            "canal": canal,
            "servicio": servicio,
            "fecha_inicio": inicio.isoformat(),
            "fecha_fin": fin.isoformat(),
            "fecha_hora": _formato_fecha_hora_calendario(inicio),
            "metadata": {
                "conversation_id": str(datos.get("conversation_id", "")),
                **metadata,
            },
        }

    def registrarAgendamientoPendiente(
        self, datos: Dict[str, Any], error_google: str
    ) -> Dict[str, Any]:
        """Guarda la solicitud como pendiente cuando Google no permite validar."""
        cita = self.validarDatosAgendamiento(datos)
        if not cita.get("servicio"):
            raise ValueError("La cita requiere el servicio cotizado.")
        metadata = datos.get("metadata", {})
        datos_cliente = metadata.get("datos_cliente") if isinstance(metadata, dict) else None
        if not isinstance(datos_cliente, dict):
            raise ValueError("Faltan los datos completos del cliente para la reserva.")
        cita["metadata"]["datos_cliente"] = validar_datos_cliente_reserva(
            datos_cliente.get("nombre_cliente", cita["nombre_cliente"]),
            datos_cliente.get("cedula"),
            datos_cliente.get("telefono"),
        )
        registro = self.repositorio.registrar_agendamiento(
            {
                **cita,
                "estado": "pendiente",
                "registrado_en": "solo_supabase",
                "google_calendar_estado": "error",
                "google_calendar_error": error_google,
                "observaciones": (
                    "Pendiente de confirmación: no se pudo validar la disponibilidad "
                    "en Google Calendar."
                ),
            }
        )
        return {"status": "pendiente", **registro}

    def registrarCitaEnCalendario(
        self,
        fecha_hora: str,
        cliente: str,
        estado: str = "OCUPADO",
        id_cliente: str = "cliente_001",
        canal: str = "WhatsApp",
        servicio: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persiste la cita y confirma solo después de completar las escrituras."""
        base = self.validarDatosAgendamiento(
            {
                "fecha_hora": fecha_hora,
                "id_cliente": id_cliente,
                "nombre_cliente": cliente,
                "canal": canal,
                "servicio": servicio,
                "metadata": metadata or {},
            }
        )
        if not base.get("servicio"):
            raise ValueError("La cita requiere el servicio cotizado.")
        metadata_inicial = metadata or {}
        datos_cliente_inicial = metadata_inicial.get("datos_cliente")
        if not isinstance(datos_cliente_inicial, dict):
            raise ValueError("Faltan los datos completos del cliente para la reserva.")
        datos_cliente_validos = validar_datos_cliente_reserva(
            datos_cliente_inicial.get("nombre_cliente", base["nombre_cliente"]),
            datos_cliente_inicial.get("cedula"),
            datos_cliente_inicial.get("telefono"),
        )
        if datos_cliente_validos["nombre_cliente"] != base["nombre_cliente"]:
            raise ValueError("El nombre de la reserva debe coincidir con el dato recopilado.")
        base["metadata"]["datos_cliente"] = datos_cliente_validos
        datos_cliente_reserva = base["metadata"]["datos_cliente"]
        base["metadata"].update(
            {
                clave: valor
                for clave, valor in (metadata or {}).items()
                if clave != "datos_cliente"
            }
        )
        base["metadata"]["datos_cliente"] = datos_cliente_reserva
        base.update(
            {
                "estado": "confirmada",
                "registrado_en": "solo_supabase",
                "google_calendar_estado": "no_configurado",
                "google_event_id": None,
                "google_event_link": None,
                "google_calendar_error": None,
                "observaciones": None,
            }
        )
        google_calendar = self.agendaVirtual.google_calendar
        fecha_hora_normalizada = base["fecha_hora"]

        if google_calendar is None:
            if not self.agendaVirtual.verificar_disponibilidad(
                fecha_hora_normalizada
            ):
                raise ConflictoAgendamientoError(
                    "El horario solicitado se cruza con otra cita activa."
                )
            registro = self.repositorio.registrar_agendamiento(base)
        else:
            try:
                disponible = self.agendaVirtual.verificar_disponibilidad(
                    fecha_hora_normalizada
                )
            except ErrorGoogleCalendar as exc:
                pendiente = {
                    **base,
                    "estado": "pendiente",
                    "google_calendar_estado": "error",
                    "google_calendar_error": str(exc),
                    "observaciones": "Pendiente: no se pudo validar la disponibilidad externa.",
                }
                registro = self.repositorio.registrar_agendamiento(pendiente)
                return {"status": "pendiente", **registro}

            if not disponible:
                raise ConflictoAgendamientoError(
                    "El horario solicitado se cruza con otra cita activa."
                )

            pendiente = {
                **base,
                "estado": "pendiente",
                "google_calendar_estado": "pendiente",
                "observaciones": "Reserva en proceso de sincronización con Google Calendar.",
            }
            registro = self.repositorio.registrar_agendamiento(pendiente)
            try:
                servicio_nombre = str(
                    self.plataformas.portafolio_servicios.get(servicio, {}).get(
                        "nombre", servicio
                    )
                )
                evento = google_calendar.crear_evento(
                    fecha_hora=fecha_hora_normalizada,
                    cliente=cliente,
                    titulo=f"{servicio_nombre} — {cliente}",
                    descripcion=(
                        f"Servicio: {servicio_nombre}\n"
                        f"Cliente: {cliente}\n"
                        f"Teléfono: {datos_cliente_reserva['telefono']}"
                    ),
                )
            except ErrorGoogleCalendar as exc:
                registro = self.repositorio.actualizar_agendamiento(
                    str(registro["id"]),
                    {
                        "estado": "confirmada",
                        "registrado_en": "solo_supabase",
                        "google_calendar_estado": "error",
                        "google_calendar_error": str(exc),
                        "observaciones": "Cita guardada en Supabase; Google Calendar no sincronizó.",
                    },
                )
            else:
                try:
                    registro = self.repositorio.actualizar_agendamiento(
                        str(registro["id"]),
                        {
                            "estado": "confirmada",
                            "registrado_en": "supabase_y_google_calendar",
                            "google_calendar_estado": "sincronizado",
                            "google_event_id": evento["evento_google_id"],
                            "google_event_link": evento.get("html_link"),
                            "google_calendar_error": None,
                            "observaciones": "Cita sincronizada con Google Calendar.",
                        },
                    )
                except (
                    httpx.HTTPError,
                    sqlite3.Error,
                    RuntimeError,
                    ValueError,
                ) as error_actualizacion:
                    try:
                        estado_persistido = self.repositorio.obtener_agendamiento(
                            str(registro["id"])
                        )
                    except (httpx.HTTPError, sqlite3.Error, RuntimeError, ValueError):
                        estado_persistido = None

                    if (
                        estado_persistido
                        and estado_persistido.get("estado") == "confirmada"
                        and estado_persistido.get("google_calendar_estado")
                        == "sincronizado"
                        and estado_persistido.get("google_event_id")
                        == evento["evento_google_id"]
                    ):
                        registro = estado_persistido
                    else:
                        try:
                            registro = self.repositorio.actualizar_agendamiento(
                                str(registro["id"]),
                                {
                                    "estado": "pendiente",
                                    "registrado_en": "solo_supabase",
                                    "google_calendar_estado": "error",
                                    "google_event_id": evento["evento_google_id"],
                                    "google_event_link": evento.get("html_link"),
                                    "google_calendar_error": (
                                        "Se creó el evento de Google Calendar, pero no se "
                                        "pudo verificar su sincronización en Supabase "
                                        f"({type(error_actualizacion).__name__})."
                                    ),
                                    "observaciones": (
                                        "Pendiente de reconciliación manual. No se eliminó "
                                        "el evento y no se debe reintentar la sincronización "
                                        "sin verificar primero su identificador."
                                    ),
                                },
                            )
                            return {"status": "pendiente", **registro}
                        except (
                            httpx.HTTPError,
                            sqlite3.Error,
                            RuntimeError,
                            ValueError,
                        ) as error_registro:
                            self._emitir_traza(
                                "EntornoVirtualAtencion",
                                "No se pudo persistir la recuperación de sincronización "
                                f"para evento Google {evento['evento_google_id']} "
                                f"({type(error_registro).__name__}).",
                            )
                            raise error_actualizacion from error_registro

        if registro.get("estado") != "confirmada":
            raise RuntimeError(
                "La persistencia no devolvió un agendamiento confirmado."
            )

        registro_local = {
            **registro,
            "estado": "OCUPADO",
            "cliente": cliente,
        }
        self.agendaVirtual.bloques[fecha_hora_normalizada] = registro_local
        self._emitir_traza(
            "AgenteAgendamientos -> EntornoVirtualAtencion",
            f"administrarCitaYNotificar({fecha_hora_normalizada}, {estado}) -> persistido({registro['registrado_en']})",
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
            "pausarYAlertarHumano() -> ok(alerta_recibida_por_profesional)",
        )
        return alerta
