"""
Repositorio multi-negocio (multi-tenant) para la plataforma comercial.
Garantiza aislamiento estricto por `negocio_id` en Supabase PostgreSQL (REST) y SQLite.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx


HORARIO_DEFAULT: Dict[str, Any] = {
    "dias": [1, 2, 3, 4, 5, 6],
    "inicio": "08:00",
    "fin": "18:00",
    "duracion_min": 60,
}

_RE_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
_NOMBRES_PLACEHOLDER = {"", "Cliente", "Webchat Cliente", "WhatsApp Cliente"}


class ConflictoAgendaError(ValueError):
    """Error lanzado cuando dos citas activas de un mismo negocio se cruzan."""


class ConflictoCanalError(ValueError):
    """Error lanzado cuando un numero/ID de WhatsApp o Botpress ya pertenece a otro negocio."""


def normalizar_slug(valor: str) -> str:
    """Convierte un nombre en un identificador seguro snake_case."""
    limpio = re.sub(r"[^a-zA-Z0-9]+", "_", (valor or "").strip().lower()).strip("_")
    return limpio or f"negocio_{uuid.uuid4().hex[:6]}"


def normalizar_telefono(valor: str) -> str:
    """Retorna solo digitos de un numero telefonico (para enrutamiento y wa.me)."""
    return re.sub(r"\D+", "", str(valor or ""))


def resolver_zona_horaria_segura(nombre_tz: str, fallback: str = "America/Bogota") -> ZoneInfo:
    """Resuelve una zona horaria IANA con respaldo seguro."""
    candidato = str(nombre_tz or "").strip() or fallback
    try:
        return ZoneInfo(candidato)
    except ZoneInfoNotFoundError:
        return ZoneInfo(fallback)


def validar_zona_horaria(nombre_tz: str) -> str:
    """Valida que la zona horaria IANA exista; lanza ValueError si es invalida."""
    limpio = str(nombre_tz or "").strip() or "America/Bogota"
    try:
        ZoneInfo(limpio)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Zona horaria invalida: '{limpio}'.") from exc
    return limpio


def _minutos_hhmm(valor: str) -> Optional[int]:
    m = _RE_HHMM.match(str(valor or ""))
    if not m:
        return None
    h, mn = int(m.group(1)), int(m.group(2))
    if 0 <= h <= 23 and 0 <= mn <= 59:
        return h * 60 + mn
    return None


def validar_horario_laboral(horario: Dict[str, Any]) -> Dict[str, Any]:
    """Valida y normaliza un objeto horario_laboral (dias 1..7 e inicio < fin)."""
    if not isinstance(horario, dict):
        raise ValueError("horario_laboral debe ser un objeto JSON.")
    dias_raw = horario.get("dias", HORARIO_DEFAULT["dias"])
    if not isinstance(dias_raw, list):
        raise ValueError("horario_laboral.dias debe ser una lista de enteros entre 1 y 7.")
    dias = sorted({int(d) for d in dias_raw if str(d).isdigit() and 1 <= int(d) <= 7})
    if not dias:
        raise ValueError("horario_laboral.dias debe incluir al menos un dia valido (1=Lunes..7=Domingo).")

    inicio = str(horario.get("inicio") or HORARIO_DEFAULT["inicio"]).strip()
    fin = str(horario.get("fin") or HORARIO_DEFAULT["fin"]).strip()
    min_ini = _minutos_hhmm(inicio)
    min_fin = _minutos_hhmm(fin)
    if min_ini is None or min_fin is None:
        raise ValueError("Las horas de apertura y cierre deben usar formato HH:MM.")
    if min_fin <= min_ini:
        raise ValueError("La hora de cierre debe ser posterior a la hora de apertura.")

    duracion = int(horario.get("duracion_min") or HORARIO_DEFAULT["duracion_min"])
    if not 15 <= duracion <= 720:
        raise ValueError("duracion_min debe estar entre 15 y 720 minutos.")

    return {
        "dias": dias,
        "inicio": f"{min_ini // 60:02d}:{min_ini % 60:02d}",
        "fin": f"{min_fin // 60:02d}:{min_fin % 60:02d}",
        "duracion_min": duracion,
    }


def _es_uuid_valido(valor: str) -> bool:
    try:
        uuid.UUID(str(valor or ""))
        return True
    except (ValueError, TypeError):
        return False


def hashear_pin(pin: str) -> str:
    """Genera hash PBKDF2-HMAC-SHA256 con sal aleatoria para el PIN del dueno."""
    pin_limpio = (pin or "").strip()
    if not pin_limpio:
        return ""
    sal = secrets.token_hex(16)
    derivado = hashlib.pbkdf2_hmac(
        "sha256", pin_limpio.encode("utf-8"), sal.encode("utf-8"), 120_000
    ).hex()
    return f"pbkdf2_sha256${sal}${derivado}"


def verificar_pin(pin: str, hash_guardado: str) -> bool:
    """Valida en tiempo constante el PIN del dueno contra el hash almacenado."""
    if not pin or not hash_guardado or "$" not in hash_guardado:
        return False
    partes = hash_guardado.split("$")
    if len(partes) != 3 or partes[0] != "pbkdf2_sha256":
        return False
    _, sal, esperado = partes
    calculado = hashlib.pbkdf2_hmac(
        "sha256", pin.strip().encode("utf-8"), sal.encode("utf-8"), 120_000
    ).hex()
    return hmac.compare_digest(calculado, esperado)


def pin_fingerprint(hash_guardado: str) -> str:
    """Huella corta del hash de PIN para invalidar sesiones previas al rotar PIN."""
    if not hash_guardado:
        return ""
    return hashlib.sha256(hash_guardado.encode("utf-8")).hexdigest()[:16]


def generar_token_sesion(
    secreto: str,
    rol: str,
    negocio_id: Optional[str] = None,
    ttl_segundos: int = 86_400 * 7,
    pin_fp: str = "",
) -> str:
    """Genera un token firmado HMAC-SHA256 con expiracion (sin dependencias externas)."""
    if not secreto or not secreto.strip():
        raise ValueError("SESSION_SECRET no puede estar vacio al firmar tokens.")
    carga: Dict[str, Any] = {
        "rol": rol,
        "negocio_id": negocio_id,
        "exp": int(time.time()) + ttl_segundos,
    }
    if pin_fp:
        carga["pin_fp"] = pin_fp
    raw = base64.urlsafe_b64encode(
        json.dumps(carga, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    firma = hmac.new(
        secreto.encode("utf-8"), raw.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{raw}.{firma}"


def validar_token_sesion(secreto: str, token: str) -> Optional[Dict[str, Any]]:
    """Verifica firma y vigencia de un token de sesion."""
    if not secreto or not secreto.strip() or not token or "." not in token:
        return None
    raw, firma = token.rsplit(".", 1)
    esperada = hmac.new(
        secreto.encode("utf-8"), raw.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(firma, esperada):
        return None
    try:
        datos = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
        if int(datos.get("exp", 0)) < int(time.time()):
            return None
        return datos
    except Exception:
        return None


def _parse_json_seguro(valor: Any, default: Any) -> Any:
    if isinstance(valor, type(default)):
        return valor
    if isinstance(valor, str):
        try:
            cargado = json.loads(valor)
            if isinstance(cargado, type(default)):
                return cargado
        except Exception:
            pass
    return default


def _normalizar_negocio(fila: Dict[str, Any]) -> Dict[str, Any]:
    horario_raw = _parse_json_seguro(fila.get("horario_laboral"), dict(HORARIO_DEFAULT))
    try:
        horario_norm = validar_horario_laboral(horario_raw) if horario_raw else dict(HORARIO_DEFAULT)
    except ValueError:
        horario_norm = dict(HORARIO_DEFAULT)

    zona_raw = str(fila.get("zona_horaria") or "America/Bogota").strip()
    try:
        zona_norm = validar_zona_horaria(zona_raw)
    except ValueError:
        zona_norm = "America/Bogota"

    reglas = _parse_json_seguro(fila.get("reglas_y_faq"), {})
    return {
        "id": str(fila["id"]),
        "nombre_negocio": str(fila.get("nombre_negocio", "")),
        "rubro": str(fila.get("rubro", "")),
        "moneda": str(fila.get("moneda") or "COP"),
        "unidad_cantidad": str(fila.get("unidad_cantidad") or "servicios"),
        "etiqueta_cantidad": str(fila.get("etiqueta_cantidad") or "cantidad"),
        "etiqueta_complejidad": str(fila.get("etiqueta_complejidad") or "complejidad"),
        "telefono_dueno": str(fila.get("telefono_dueno") or "").strip(),
        "whatsapp_negocio": normalizar_telefono(str(fila.get("whatsapp_negocio") or "")),
        "whatsapp_phone_number_id": str(fila.get("whatsapp_phone_number_id") or "").strip(),
        "whatsapp_access_token": str(fila.get("whatsapp_access_token") or "").strip(),
        "botpress_bot_id": str(fila.get("botpress_bot_id") or "").strip(),
        "google_calendar_id": str(fila.get("google_calendar_id") or "").strip(),
        "zona_horaria": zona_norm,
        "margen_base": float(fila.get("margen_base", 0.35)),
        "margen_minimo": float(fila.get("margen_minimo", 0.22)),
        "descuento_maximo": float(fila.get("descuento_maximo", 15.0)),
        "mensaje_catalogo_intro": str(
            fila.get("mensaje_catalogo_intro") or "Con gusto te comparto nuestros servicios"
        ),
        "activo": bool(fila.get("activo", True)),
        "horario_laboral": horario_norm,
        "reglas_y_faq": reglas,
        "pin_dueno_hash": str(fila.get("pin_dueno_hash") or ""),
        "tiene_pin_dueno": bool(fila.get("pin_dueno_hash")),
    }


def sanitizar_negocio_publico(negocio: Dict[str, Any]) -> Dict[str, Any]:
    """Oculta secretos (PIN hash y access token) antes de responder por API."""
    copia = dict(negocio)
    copia.pop("pin_dueno_hash", None)
    token = str(copia.pop("whatsapp_access_token", "") or "")
    copia["tiene_whatsapp_token"] = bool(token)
    return copia


def _semillas_negocios() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Semillas iniciales listas para Fotografia y Salon de Belleza."""
    negocios = [
        {
            "id": "estudio_fotografia",
            "nombre_negocio": "Estudio Visual Independiente",
            "rubro": "Fotografia Comercial y Produccion de Video",
            "moneda": "COP",
            "unidad_cantidad": "sesiones",
            "etiqueta_cantidad": "sesiones_requeridas",
            "etiqueta_complejidad": "complejidad",
            "telefono_dueno": "",
            "whatsapp_negocio": "",
            "whatsapp_phone_number_id": "",
            "whatsapp_access_token": "",
            "botpress_bot_id": "",
            "google_calendar_id": "",
            "zona_horaria": "America/Bogota",
            "margen_base": 0.40,
            "margen_minimo": 0.25,
            "descuento_maximo": 12.0,
            "mensaje_catalogo_intro": "Hola! Aqui tienes nuestros paquetes de fotografia y produccion visual",
            "activo": True,
            "horario_laboral": {
                "dias": [1, 2, 3, 4, 5, 6],
                "inicio": "08:00",
                "fin": "18:00",
                "duracion_min": 60,
            },
            "reglas_y_faq": {
                "tono_marca": "Cercano, profesional y claro. Guia al cliente sin presionar.",
                "direccion": "Estudio privado (confirmar ubicacion exacta al agendar).",
                "metodos_pago": "Transferencia bancaria, Nequi y efectivo (50% al reservar y 50% contra entrega).",
                "politicas": "Reprogramacion gratuita con 24 horas de anticipacion. Entrega digital en 5 dias habiles.",
                "preguntas_frecuentes": [
                    {
                        "pregunta": "Las fotos se entregan editadas?",
                        "respuesta": "Si, todas las fotos del paquete se entregan editadas en alta resolucion en galeria digital.",
                    }
                ],
            },
            "pin_dueno_hash": hashear_pin("1234"),
        },
        {
            "id": "salon_belleza",
            "nombre_negocio": "Atelier Belleza y Estilo",
            "rubro": "Salon de Belleza, Cabello, Unas y Cejas",
            "moneda": "COP",
            "unidad_cantidad": "servicios",
            "etiqueta_cantidad": "cantidad_servicios",
            "etiqueta_complejidad": "complejidad",
            "telefono_dueno": "",
            "whatsapp_negocio": "",
            "whatsapp_phone_number_id": "",
            "whatsapp_access_token": "",
            "botpress_bot_id": "",
            "google_calendar_id": "",
            "zona_horaria": "America/Bogota",
            "margen_base": 0.45,
            "margen_minimo": 0.25,
            "descuento_maximo": 10.0,
            "mensaje_catalogo_intro": "Hola! Con gusto te comparto nuestros servicios de belleza y cuidado personal",
            "activo": True,
            "horario_laboral": {
                "dias": [2, 3, 4, 5, 6],
                "inicio": "09:00",
                "fin": "19:00",
                "duracion_min": 60,
            },
            "reglas_y_faq": {
                "tono_marca": "Calido, amable y atento, ideal para atencion en salon de belleza.",
                "direccion": "Sede principal (actualiza tu direccion exacta desde el panel del negocio).",
                "metodos_pago": "Nequi, Daviplata, transferencia bancaria y efectivo en el salon.",
                "politicas": "Llegar 10 minutos antes de la cita. Cambios o cancelaciones con minimo 4 horas de anticipacion.",
                "preguntas_frecuentes": [
                    {
                        "pregunta": "Atienden los lunes?",
                        "respuesta": "Nuestro horario habitual es de martes a sabado de 9:00 a. m. a 7:00 p. m.",
                    }
                ],
            },
            "pin_dueno_hash": hashear_pin("1234"),
        },
    ]

    servicios = [
        # Fotografia
        {
            "negocio_id": "estudio_fotografia",
            "clave": "sesion_producto",
            "servicio_id_num": 1,
            "nombre": "Fotografia de Producto E-Commerce",
            "descripcion": "Incluye 10 fotos editadas en alta resolucion en estudio, iluminacion profesional y fondo limpio.",
            "tarifa_unitaria": 280000.0,
            "palabras_clave": ["producto", "catalogo", "articulo", "ecommerce"],
            "duracion_minutos": 60,
            "activo": True,
        },
        {
            "negocio_id": "estudio_fotografia",
            "clave": "retrato_corporativo",
            "servicio_id_num": 2,
            "nombre": "Retrato Corporativo y Marca Personal",
            "descripcion": "Incluye 8 fotos editadas en alta resolucion en estudio (hasta 2 personas) con retoque profesional.",
            "tarifa_unitaria": 320000.0,
            "palabras_clave": ["retrato", "corporativo", "perfil", "marca personal"],
            "duracion_minutos": 60,
            "activo": True,
        },
        {
            "negocio_id": "estudio_fotografia",
            "clave": "cobertura_evento",
            "servicio_id_num": 3,
            "nombre": "Cobertura Fotografica de Eventos",
            "descripcion": "Incluye 30 fotos editadas en galeria digital (2 a 3 horas en locacion).",
            "tarifa_unitaria": 450000.0,
            "palabras_clave": ["evento", "boda", "grado", "fiesta"],
            "duracion_minutos": 180,
            "activo": True,
        },
        {
            "negocio_id": "estudio_fotografia",
            "clave": "video_reel",
            "servicio_id_num": 4,
            "nombre": "Produccion de Video Reels / Comercial Corto",
            "descripcion": "Incluye 2 reels 4K editados, colorizacion y musica para redes sociales.",
            "tarifa_unitaria": 390000.0,
            "palabras_clave": ["video", "reel", "reels", "comercial"],
            "duracion_minutos": 120,
            "activo": True,
        },
        # Belleza
        {
            "negocio_id": "salon_belleza",
            "clave": "manicure_semipermanente",
            "servicio_id_num": 1,
            "nombre": "Manicure Semipermanente",
            "descripcion": "Limpieza completa, hidratacion de cuticula y esmaltado semipermanente de larga duracion.",
            "tarifa_unitaria": 45000.0,
            "palabras_clave": ["manicure", "unas", "semipermanente", "manos"],
            "duracion_minutos": 60,
            "activo": True,
        },
        {
            "negocio_id": "salon_belleza",
            "clave": "pedicure_spa",
            "servicio_id_num": 2,
            "nombre": "Pedicure Spa Completo",
            "descripcion": "Exfoliacion, limpieza profunda, hidratacion spa y esmaltado tradicional o semipermanente.",
            "tarifa_unitaria": 55000.0,
            "palabras_clave": ["pedicure", "pies", "spa"],
            "duracion_minutos": 60,
            "activo": True,
        },
        {
            "negocio_id": "salon_belleza",
            "clave": "diseno_cejas_laminado",
            "servicio_id_num": 3,
            "nombre": "Diseno de Cejas y Laminado",
            "descripcion": "Diseno segun morfologia del rostro, depilacion con hilo o cera y laminado nutritivo.",
            "tarifa_unitaria": 65000.0,
            "palabras_clave": ["cejas", "laminado", "depilacion", "pestanas"],
            "duracion_minutos": 45,
            "activo": True,
        },
        {
            "negocio_id": "salon_belleza",
            "clave": "cepillado_hidratacion",
            "servicio_id_num": 4,
            "nombre": "Cepillado y Tratamiento de Hidratacion",
            "descripcion": "Lavado detox, mascarilla nutritiva de keratina o colageno y cepillado profesional.",
            "tarifa_unitaria": 75000.0,
            "palabras_clave": ["cepillado", "hidratacion", "cabello", "pelo"],
            "duracion_minutos": 90,
            "activo": True,
        },
        {
            "negocio_id": "salon_belleza",
            "clave": "balayage_iluminacion",
            "servicio_id_num": 5,
            "nombre": "Balayage e Iluminacion Personalizada",
            "descripcion": "Diseno de color a medida con proteccion capilar, matizacion y finalizacion con ondas.",
            "tarifa_unitaria": 260000.0,
            "palabras_clave": ["balayage", "color", "tintura", "iluminaciones"],
            "duracion_minutos": 180,
            "activo": True,
        },
    ]

    promociones = [
        {
            "negocio_id": "estudio_fotografia",
            "codigo": "PACK_SESIONES_10",
            "descripcion": "10% de descuento al reservar 2 o mas sesiones",
            "descuento_max": 10.0,
            "activa": True,
            "condiciones": {"cantidad_minima": 2, "cantidad_maxima": 20},
        },
        {
            "negocio_id": "salon_belleza",
            "codigo": "COMBO_BELLEZA_10",
            "descripcion": "10% de descuento al reservar 2 o mas servicios en la misma visita",
            "descuento_max": 10.0,
            "activa": True,
            "condiciones": {"cantidad_minima": 2, "cantidad_maxima": 10},
        },
    ]
    return negocios, servicios, promociones


class RepositorioMultiTenant(ABC):
    """Contrato unificado multi-negocio para SQLite y Supabase."""

    modo_bd: str = "desconocido"

    @abstractmethod
    def listar_negocios(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def obtener_negocio(self, negocio_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def guardar_negocio(self, datos: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def eliminar_negocio(self, negocio_id: str) -> bool: ...

    def _validar_colision_canales(self, negocio_id: str, norm: Dict[str, Any]) -> None:
        """Impide que dos negocios compartan el mismo Phone Number ID, WhatsApp o Botpress ID."""
        pnid = norm.get("whatsapp_phone_number_id", "")
        wa_num = normalizar_telefono(norm.get("whatsapp_negocio", ""))
        bp_id = norm.get("botpress_bot_id", "")
        if not pnid and not wa_num and not bp_id:
            return
        for otro in self.listar_negocios():
            if otro["id"] == negocio_id:
                continue
            if pnid and otro.get("whatsapp_phone_number_id") == pnid:
                raise ConflictoCanalError(
                    f"El Meta Phone Number ID '{pnid}' ya esta asignado a otro negocio."
                )
            if wa_num and normalizar_telefono(otro.get("whatsapp_negocio", "")) == wa_num:
                raise ConflictoCanalError(
                    f"El numero de WhatsApp '{wa_num}' ya esta asignado a otro negocio."
                )
            if bp_id and otro.get("botpress_bot_id") == bp_id:
                raise ConflictoCanalError(
                    f"El botpress_bot_id '{bp_id}' ya esta asignado a otro negocio."
                )

    def buscar_negocio_por_canal(
        self,
        phone_number_id: str = "",
        whatsapp_negocio: str = "",
        botpress_bot_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Enruta automaticamente un webhook entrante al negocio exacto."""
        pnid = (phone_number_id or "").strip()
        wa_num = normalizar_telefono(whatsapp_negocio)
        bp_id = (botpress_bot_id or "").strip()
        for neg in self.listar_negocios():
            if not neg.get("activo", True):
                continue
            if pnid and neg.get("whatsapp_phone_number_id") == pnid:
                return neg
            if bp_id and neg.get("botpress_bot_id") == bp_id:
                return neg
            if wa_num and normalizar_telefono(neg.get("whatsapp_negocio", "")) == wa_num:
                return neg
        return None

    def _tz_negocio(self, negocio_id: str) -> ZoneInfo:
        neg = self.obtener_negocio(negocio_id)
        return resolver_zona_horaria_segura((neg or {}).get("zona_horaria", "America/Bogota"))

    @abstractmethod
    def listar_servicios(self, negocio_id: str, solo_activos: bool = False) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def guardar_servicio(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def eliminar_servicio(self, negocio_id: str, clave: str) -> bool: ...

    @abstractmethod
    def listar_promociones(self, negocio_id: str, solo_activas: bool = False) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def guardar_promocion(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def eliminar_promocion(self, negocio_id: str, codigo: str) -> bool: ...

    @abstractmethod
    def obtener_perfil(self, negocio_id: str, id_cliente: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def obtener_o_crear_perfil(
        self, negocio_id: str, id_cliente: str, nombre: str = "Cliente", canal: str = "WhatsApp"
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def guardar_perfil(self, negocio_id: str, perfil: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def listar_perfiles(self, negocio_id: str, solo_pausados: bool = False) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def registrar_alerta(self, negocio_id: str, alerta: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def listar_alertas(self, negocio_id: str, solo_abiertas: bool = False) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def resolver_alertas_cliente(self, negocio_id: str, id_cliente: str) -> None: ...

    @abstractmethod
    def listar_agendamientos(
        self,
        negocio_id: str,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        solo_activos: bool = False,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def obtener_agendamiento(self, negocio_id: str, id_agendamiento: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def registrar_agendamiento(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def actualizar_agendamiento(
        self, negocio_id: str, id_agendamiento: str, cambios: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def cancelar_agendamiento(self, negocio_id: str, id_agendamiento: str) -> bool: ...


class RepositorioComercialSQLite(RepositorioMultiTenant):
    """Implementacion SQLite multi-negocio thread-safe para desarrollo local y pruebas TDD."""

    modo_bd = "sqlite_multitenant"

    def __init__(self, ruta_db: str = "data/comercial_multitenant.db") -> None:
        self.ruta_db = ruta_db
        self._lock = threading.RLock()
        if ruta_db != ":memory:":
            Path(ruta_db).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(ruta_db, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._inicializar_esquema()
        self._sembrar_si_vacio()

    def _inicializar_esquema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS negocios (
                    id TEXT PRIMARY KEY,
                    nombre_negocio TEXT NOT NULL,
                    rubro TEXT NOT NULL,
                    moneda TEXT NOT NULL DEFAULT 'COP',
                    unidad_cantidad TEXT NOT NULL DEFAULT 'servicios',
                    etiqueta_cantidad TEXT NOT NULL DEFAULT 'cantidad',
                    etiqueta_complejidad TEXT NOT NULL DEFAULT 'complejidad',
                    telefono_dueno TEXT NOT NULL DEFAULT '',
                    whatsapp_negocio TEXT NOT NULL DEFAULT '',
                    whatsapp_phone_number_id TEXT NOT NULL DEFAULT '',
                    whatsapp_access_token TEXT NOT NULL DEFAULT '',
                    botpress_bot_id TEXT NOT NULL DEFAULT '',
                    google_calendar_id TEXT NOT NULL DEFAULT '',
                    zona_horaria TEXT NOT NULL DEFAULT 'America/Bogota',
                    margen_base REAL NOT NULL DEFAULT 0.35,
                    margen_minimo REAL NOT NULL DEFAULT 0.22,
                    descuento_maximo REAL NOT NULL DEFAULT 15.0,
                    mensaje_catalogo_intro TEXT NOT NULL DEFAULT '',
                    activo INTEGER NOT NULL DEFAULT 1,
                    horario_laboral TEXT NOT NULL DEFAULT '{}',
                    reglas_y_faq TEXT NOT NULL DEFAULT '{}',
                    pin_dueno_hash TEXT NOT NULL DEFAULT ''
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_wa_phone_id
                    ON negocios (whatsapp_phone_number_id)
                    WHERE whatsapp_phone_number_id <> '';

                CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_wa_numero
                    ON negocios (whatsapp_negocio)
                    WHERE whatsapp_negocio <> '';

                CREATE UNIQUE INDEX IF NOT EXISTS idx_negocios_botpress_id
                    ON negocios (botpress_bot_id)
                    WHERE botpress_bot_id <> '';

                CREATE TABLE IF NOT EXISTS servicios (
                    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
                    clave TEXT NOT NULL,
                    servicio_id_num INTEGER NOT NULL DEFAULT 1,
                    nombre TEXT NOT NULL,
                    descripcion TEXT NOT NULL DEFAULT '',
                    tarifa_unitaria REAL NOT NULL,
                    palabras_clave TEXT NOT NULL DEFAULT '[]',
                    requiere_complejidad INTEGER NOT NULL DEFAULT 0,
                    costo_adicional_mantenimiento REAL NOT NULL DEFAULT 0.0,
                    duracion_minutos INTEGER NOT NULL DEFAULT 60,
                    activo INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (negocio_id, clave)
                );

                CREATE TABLE IF NOT EXISTS promociones_vigentes (
                    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
                    codigo TEXT NOT NULL,
                    descripcion TEXT NOT NULL,
                    descuento_max REAL NOT NULL,
                    activa INTEGER NOT NULL DEFAULT 1,
                    condiciones TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (negocio_id, codigo)
                );

                CREATE TABLE IF NOT EXISTS perfiles_clientes (
                    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
                    id_cliente TEXT NOT NULL,
                    nombre TEXT NOT NULL,
                    canal TEXT NOT NULL,
                    historial_interacciones TEXT NOT NULL DEFAULT '[]',
                    tono_actual TEXT NOT NULL DEFAULT 'neutral',
                    fidelidad INTEGER NOT NULL DEFAULT 1,
                    ultima_cotizacion REAL,
                    parametros_parciales TEXT NOT NULL DEFAULT '{}',
                    automatizacion_activa INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (negocio_id, id_cliente)
                );

                CREATE TABLE IF NOT EXISTS alertas_humanas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
                    timestamp TEXT NOT NULL,
                    id_cliente TEXT NOT NULL,
                    nombre TEXT NOT NULL,
                    canal TEXT NOT NULL,
                    motivo_critico TEXT NOT NULL,
                    resumen_ia TEXT NOT NULL DEFAULT '',
                    historial_charla TEXT NOT NULL DEFAULT '[]',
                    estado TEXT NOT NULL DEFAULT 'pausado',
                    resuelta INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS agendamientos (
                    id TEXT PRIMARY KEY,
                    negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
                    id_cliente TEXT NOT NULL,
                    nombre_cliente TEXT NOT NULL,
                    canal TEXT NOT NULL,
                    servicio TEXT NOT NULL DEFAULT '',
                    fecha_inicio TEXT NOT NULL,
                    fecha_fin TEXT NOT NULL,
                    estado TEXT NOT NULL DEFAULT 'confirmada',
                    registrado_en TEXT NOT NULL DEFAULT 'solo_supabase',
                    google_calendar_estado TEXT NOT NULL DEFAULT 'no_configurado',
                    google_event_id TEXT,
                    google_event_link TEXT,
                    observaciones TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_agend_negocio_fecha
                    ON agendamientos (negocio_id, fecha_inicio, fecha_fin);
                """
            )

    def _sembrar_si_vacio(self) -> None:
        with self._lock:
            conteo = self._conn.execute("SELECT COUNT(*) FROM negocios").fetchone()[0]
            if conteo > 0:
                return
            negocios, servicios, promociones = _semillas_negocios()
            for neg in negocios:
                self.guardar_negocio(neg)
            for srv in servicios:
                self.guardar_servicio(srv["negocio_id"], srv)
            for promo in promociones:
                self.guardar_promocion(promo["negocio_id"], promo)

    def listar_negocios(self) -> List[Dict[str, Any]]:
        with self._lock:
            filas = self._conn.execute("SELECT * FROM negocios ORDER BY nombre_negocio ASC").fetchall()
            return [_normalizar_negocio(dict(r)) for r in filas]

    def obtener_negocio(self, negocio_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            fila = self._conn.execute("SELECT * FROM negocios WHERE id = ?", (negocio_id,)).fetchone()
            return _normalizar_negocio(dict(fila)) if fila else None

    def guardar_negocio(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            neg_id = normalizar_slug(str(datos.get("id") or datos.get("nombre_negocio") or ""))
            existente = self.obtener_negocio(neg_id)
            combinado = {**(existente or {}), **datos, "id": neg_id}
            if datos.get("pin_dueno"):
                combinado["pin_dueno_hash"] = hashear_pin(str(datos["pin_dueno"]))
            elif not combinado.get("pin_dueno_hash"):
                combinado["pin_dueno_hash"] = (existente or {}).get("pin_dueno_hash", "")
            if "whatsapp_access_token" not in datos and existente:
                combinado["whatsapp_access_token"] = existente.get("whatsapp_access_token", "")

            norm = _normalizar_negocio(combinado)
            self._validar_colision_canales(neg_id, norm)
            try:
                self._conn.execute(
                    """
                    INSERT INTO negocios (
                        id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
                        etiqueta_complejidad, telefono_dueno, whatsapp_negocio,
                        whatsapp_phone_number_id, whatsapp_access_token, botpress_bot_id,
                        google_calendar_id, zona_horaria, margen_base, margen_minimo,
                        descuento_maximo, mensaje_catalogo_intro, activo,
                        horario_laboral, reglas_y_faq, pin_dueno_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        nombre_negocio=excluded.nombre_negocio,
                        rubro=excluded.rubro,
                        moneda=excluded.moneda,
                        unidad_cantidad=excluded.unidad_cantidad,
                        etiqueta_cantidad=excluded.etiqueta_cantidad,
                        etiqueta_complejidad=excluded.etiqueta_complejidad,
                        telefono_dueno=excluded.telefono_dueno,
                        whatsapp_negocio=excluded.whatsapp_negocio,
                        whatsapp_phone_number_id=excluded.whatsapp_phone_number_id,
                        whatsapp_access_token=excluded.whatsapp_access_token,
                        botpress_bot_id=excluded.botpress_bot_id,
                        google_calendar_id=excluded.google_calendar_id,
                        zona_horaria=excluded.zona_horaria,
                        margen_base=excluded.margen_base,
                        margen_minimo=excluded.margen_minimo,
                        descuento_maximo=excluded.descuento_maximo,
                        mensaje_catalogo_intro=excluded.mensaje_catalogo_intro,
                        activo=excluded.activo,
                        horario_laboral=excluded.horario_laboral,
                        reglas_y_faq=excluded.reglas_y_faq,
                        pin_dueno_hash=excluded.pin_dueno_hash
                    """,
                    (
                        norm["id"],
                        norm["nombre_negocio"],
                        norm["rubro"],
                        norm["moneda"],
                        norm["unidad_cantidad"],
                        norm["etiqueta_cantidad"],
                        norm["etiqueta_complejidad"],
                        norm["telefono_dueno"],
                        norm["whatsapp_negocio"],
                        norm["whatsapp_phone_number_id"],
                        norm["whatsapp_access_token"],
                        norm["botpress_bot_id"],
                        norm["google_calendar_id"],
                        norm["zona_horaria"],
                        norm["margen_base"],
                        norm["margen_minimo"],
                        norm["descuento_maximo"],
                        norm["mensaje_catalogo_intro"],
                        1 if norm["activo"] else 0,
                        json.dumps(norm["horario_laboral"], ensure_ascii=False),
                        json.dumps(norm["reglas_y_faq"], ensure_ascii=False),
                        norm["pin_dueno_hash"],
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictoCanalError("Ya existe otro negocio con ese canal de WhatsApp o Botpress.") from exc
            return self.obtener_negocio(neg_id) or norm

    def eliminar_negocio(self, negocio_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM negocios WHERE id = ?", (negocio_id,))
            return cur.rowcount > 0

    def listar_servicios(self, negocio_id: str, solo_activos: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM servicios WHERE negocio_id = ?"
            params: List[Any] = [negocio_id]
            if solo_activos:
                sql += " AND activo = 1"
            sql += " ORDER BY servicio_id_num ASC, nombre ASC"
            filas = self._conn.execute(sql, params).fetchall()
            salida = []
            for r in filas:
                d = dict(r)
                d["palabras_clave"] = _parse_json_seguro(d.get("palabras_clave"), [])
                d["activo"] = bool(d.get("activo", 1))
                d["duracion_minutos"] = int(d.get("duracion_minutos") or 60)
                d["tarifa_unitaria"] = float(d.get("tarifa_unitaria", 0.0))
                salida.append(d)
            return salida

    def guardar_servicio(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            clave = normalizar_slug(str(datos.get("clave") or datos.get("nombre") or ""))
            existentes = self.listar_servicios(negocio_id)
            previo = next((s for s in existentes if s["clave"] == clave), None)
            if datos.get("servicio_id_num"):
                num = int(datos["servicio_id_num"])
            elif previo:
                num = int(previo.get("servicio_id_num") or 1)
            else:
                num = max((int(s.get("servicio_id_num") or 0) for s in existentes), default=0) + 1
            palabras = datos.get("palabras_clave") or []
            if isinstance(palabras, str):
                palabras = [p.strip() for p in palabras.split(",") if p.strip()]
            self._conn.execute(
                """
                INSERT INTO servicios (
                    negocio_id, clave, servicio_id_num, nombre, descripcion,
                    tarifa_unitaria, palabras_clave, requiere_complejidad,
                    costo_adicional_mantenimiento, duracion_minutos, activo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0.0, ?, ?)
                ON CONFLICT(negocio_id, clave) DO UPDATE SET
                    nombre=excluded.nombre,
                    descripcion=excluded.descripcion,
                    tarifa_unitaria=excluded.tarifa_unitaria,
                    palabras_clave=excluded.palabras_clave,
                    duracion_minutos=excluded.duracion_minutos,
                    activo=excluded.activo
                """,
                (
                    negocio_id,
                    clave,
                    num,
                    str(datos.get("nombre") or clave),
                    str(datos.get("descripcion") or ""),
                    float(datos.get("tarifa_unitaria", 0.0)),
                    json.dumps(palabras, ensure_ascii=False),
                    max(15, int(datos.get("duracion_minutos") or 60)),
                    1 if datos.get("activo", True) else 0,
                ),
            )
            return next(s for s in self.listar_servicios(negocio_id) if s["clave"] == clave)

    def eliminar_servicio(self, negocio_id: str, clave: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM servicios WHERE negocio_id = ? AND clave = ?",
                (negocio_id, clave),
            )
            return cur.rowcount > 0

    def listar_promociones(self, negocio_id: str, solo_activas: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM promociones_vigentes WHERE negocio_id = ?"
            if solo_activas:
                sql += " AND activa = 1"
            sql += " ORDER BY codigo ASC"
            filas = self._conn.execute(sql, (negocio_id,)).fetchall()
            salida = []
            for r in filas:
                d = dict(r)
                d["activa"] = bool(d.get("activa", 1))
                d["descuento_max"] = float(d.get("descuento_max", 0.0))
                d["condiciones"] = _parse_json_seguro(d.get("condiciones"), {})
                salida.append(d)
            return salida

    def guardar_promocion(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            codigo = normalizar_slug(str(datos.get("codigo") or "PROMO")).upper()
            condiciones = _parse_json_seguro(datos.get("condiciones"), {})
            self._conn.execute(
                """
                INSERT INTO promociones_vigentes (
                    negocio_id, codigo, descripcion, descuento_max, activa, condiciones
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(negocio_id, codigo) DO UPDATE SET
                    descripcion=excluded.descripcion,
                    descuento_max=excluded.descuento_max,
                    activa=excluded.activa,
                    condiciones=excluded.condiciones
                """,
                (
                    negocio_id,
                    codigo,
                    str(datos.get("descripcion") or ""),
                    float(datos.get("descuento_max", 0.0)),
                    1 if datos.get("activa", True) else 0,
                    json.dumps(condiciones, ensure_ascii=False),
                ),
            )
            return next(p for p in self.listar_promociones(negocio_id) if p["codigo"] == codigo)

    def eliminar_promocion(self, negocio_id: str, codigo: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM promociones_vigentes WHERE negocio_id = ? AND codigo = ?",
                (negocio_id, codigo),
            )
            return cur.rowcount > 0

    def obtener_perfil(self, negocio_id: str, id_cliente: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            fila = self._conn.execute(
                "SELECT * FROM perfiles_clientes WHERE negocio_id = ? AND id_cliente = ?",
                (negocio_id, id_cliente),
            ).fetchone()
            if not fila:
                return None
            d = dict(fila)
            d["historial_interacciones"] = _parse_json_seguro(d.get("historial_interacciones"), [])
            d["parametros_parciales"] = _parse_json_seguro(d.get("parametros_parciales"), {})
            d["automatizacion_activa"] = bool(d.get("automatizacion_activa", 1))
            return d

    def obtener_o_crear_perfil(
        self, negocio_id: str, id_cliente: str, nombre: str = "Cliente", canal: str = "WhatsApp"
    ) -> Dict[str, Any]:
        with self._lock:
            existente = self.obtener_perfil(negocio_id, id_cliente)
            if existente:
                if (
                    nombre
                    and nombre not in _NOMBRES_PLACEHOLDER
                    and existente.get("nombre", "") in _NOMBRES_PLACEHOLDER
                ):
                    existente["nombre"] = nombre
                    self.guardar_perfil(negocio_id, existente)
                return existente
            nuevo = {
                "negocio_id": negocio_id,
                "id_cliente": id_cliente,
                "nombre": nombre or "Cliente",
                "canal": canal or "WhatsApp",
                "historial_interacciones": [],
                "tono_actual": "neutral",
                "fidelidad": 1,
                "ultima_cotizacion": None,
                "parametros_parciales": {},
                "automatizacion_activa": True,
            }
            return self.guardar_perfil(negocio_id, nuevo)

    def guardar_perfil(self, negocio_id: str, perfil: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            ahora = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """
                INSERT INTO perfiles_clientes (
                    negocio_id, id_cliente, nombre, canal, historial_interacciones,
                    tono_actual, fidelidad, ultima_cotizacion, parametros_parciales,
                    automatizacion_activa, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(negocio_id, id_cliente) DO UPDATE SET
                    nombre=excluded.nombre,
                    canal=excluded.canal,
                    historial_interacciones=excluded.historial_interacciones,
                    tono_actual=excluded.tono_actual,
                    fidelidad=excluded.fidelidad,
                    ultima_cotizacion=excluded.ultima_cotizacion,
                    parametros_parciales=excluded.parametros_parciales,
                    automatizacion_activa=excluded.automatizacion_activa,
                    updated_at=excluded.updated_at
                """,
                (
                    negocio_id,
                    str(perfil["id_cliente"]),
                    str(perfil.get("nombre") or "Cliente"),
                    str(perfil.get("canal") or "WhatsApp"),
                    json.dumps(perfil.get("historial_interacciones", [])[-30:], ensure_ascii=False),
                    str(perfil.get("tono_actual") or "neutral"),
                    int(perfil.get("fidelidad", 1)),
                    perfil.get("ultima_cotizacion"),
                    json.dumps(perfil.get("parametros_parciales", {}), ensure_ascii=False),
                    1 if perfil.get("automatizacion_activa", True) else 0,
                    ahora,
                ),
            )
            return {**perfil, "negocio_id": negocio_id, "updated_at": ahora}

    def listar_perfiles(self, negocio_id: str, solo_pausados: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM perfiles_clientes WHERE negocio_id = ?"
            if solo_pausados:
                sql += " AND automatizacion_activa = 0"
            sql += " ORDER BY updated_at DESC"
            filas = self._conn.execute(sql, (negocio_id,)).fetchall()
            salida = []
            for r in filas:
                d = dict(r)
                d["historial_interacciones"] = _parse_json_seguro(d.get("historial_interacciones"), [])
                d["parametros_parciales"] = _parse_json_seguro(d.get("parametros_parciales"), {})
                d["automatizacion_activa"] = bool(d.get("automatizacion_activa", 1))
                salida.append(d)
            return salida

    def registrar_alerta(self, negocio_id: str, alerta: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            tz = self._tz_negocio(negocio_id)
            ts = str(alerta.get("timestamp") or datetime.now(tz).isoformat(timespec="seconds"))
            cur = self._conn.execute(
                """
                INSERT INTO alertas_humanas (
                    negocio_id, timestamp, id_cliente, nombre, canal,
                    motivo_critico, resumen_ia, historial_charla, estado, resuelta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    negocio_id,
                    ts,
                    str(alerta["id_cliente"]),
                    str(alerta.get("nombre") or "Cliente"),
                    str(alerta.get("canal") or "WhatsApp"),
                    str(alerta.get("motivo_critico") or ""),
                    str(alerta.get("resumen_ia") or ""),
                    json.dumps(alerta.get("historial_charla", [])[-12:], ensure_ascii=False),
                    str(alerta.get("estado") or "pausado"),
                ),
            )
            return {**alerta, "id": cur.lastrowid, "negocio_id": negocio_id, "timestamp": ts, "resuelta": False}

    def listar_alertas(self, negocio_id: str, solo_abiertas: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM alertas_humanas WHERE negocio_id = ?"
            if solo_abiertas:
                sql += " AND resuelta = 0"
            sql += " ORDER BY id DESC LIMIT 100"
            filas = self._conn.execute(sql, (negocio_id,)).fetchall()
            salida = []
            for r in filas:
                d = dict(r)
                d["historial_charla"] = _parse_json_seguro(d.get("historial_charla"), [])
                d["resuelta"] = bool(d.get("resuelta", 0))
                salida.append(d)
            return salida

    def resolver_alertas_cliente(self, negocio_id: str, id_cliente: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE alertas_humanas SET resuelta = 1, estado = 'resuelta' WHERE negocio_id = ? AND id_cliente = ? AND resuelta = 0",
                (negocio_id, id_cliente),
            )

    def listar_agendamientos(
        self,
        negocio_id: str,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        solo_activos: bool = False,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM agendamientos WHERE negocio_id = ?"
            params: List[Any] = [negocio_id]
            if solo_activos:
                sql += " AND estado IN ('pendiente', 'confirmada')"
            if fecha_inicio:
                sql += " AND fecha_fin > ?"
                params.append(fecha_inicio)
            if fecha_fin:
                sql += " AND fecha_inicio < ?"
                params.append(fecha_fin)
            sql += " ORDER BY fecha_inicio ASC"
            filas = self._conn.execute(sql, params).fetchall()
            salida = []
            for r in filas:
                d = dict(r)
                d["metadata"] = _parse_json_seguro(d.get("metadata"), {})
                salida.append(d)
            return salida

    def obtener_agendamiento(self, negocio_id: str, id_agendamiento: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            fila = self._conn.execute(
                "SELECT * FROM agendamientos WHERE negocio_id = ? AND id = ?",
                (negocio_id, id_agendamiento),
            ).fetchone()
            if not fila:
                return None
            d = dict(fila)
            d["metadata"] = _parse_json_seguro(d.get("metadata"), {})
            return d

    def registrar_agendamiento(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]:
        ini_nuevo = datetime.fromisoformat(str(datos["fecha_inicio"]))
        fin_nuevo = datetime.fromisoformat(str(datos["fecha_fin"]))
        if fin_nuevo <= ini_nuevo:
            raise ValueError("La hora de fin debe ser posterior a la hora de inicio.")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                filas = self._conn.execute(
                    "SELECT fecha_inicio, fecha_fin FROM agendamientos WHERE negocio_id = ? AND estado IN ('pendiente', 'confirmada')",
                    (negocio_id,),
                ).fetchall()
                for cita in filas:
                    c_ini = datetime.fromisoformat(str(cita["fecha_inicio"]))
                    c_fin = datetime.fromisoformat(str(cita["fecha_fin"]))
                    if ini_nuevo < c_fin and fin_nuevo > c_ini:
                        raise ConflictoAgendaError("El horario solicitado se cruza con otra cita activa.")

                cita_id = str(datos.get("id") or uuid.uuid4())
                self._conn.execute(
                    """
                    INSERT INTO agendamientos (
                        id, negocio_id, id_cliente, nombre_cliente, canal, servicio,
                        fecha_inicio, fecha_fin, estado, registrado_en,
                        google_calendar_estado, google_event_id, google_event_link,
                        observaciones, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cita_id,
                        negocio_id,
                        str(datos.get("id_cliente") or ""),
                        str(datos.get("nombre_cliente") or "Cliente"),
                        str(datos.get("canal") or "WhatsApp"),
                        str(datos.get("servicio") or ""),
                        str(datos["fecha_inicio"]),
                        str(datos["fecha_fin"]),
                        str(datos.get("estado") or "confirmada"),
                        str(datos.get("registrado_en") or "solo_supabase"),
                        str(datos.get("google_calendar_estado") or "no_configurado"),
                        datos.get("google_event_id"),
                        datos.get("google_event_link"),
                        str(datos.get("observaciones") or ""),
                        json.dumps(datos.get("metadata") or {}, ensure_ascii=False),
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            return self.obtener_agendamiento(negocio_id, cita_id) or {"id": cita_id, **datos}

    def actualizar_agendamiento(
        self, negocio_id: str, id_agendamiento: str, cambios: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        permitidos = {
            "estado",
            "registrado_en",
            "google_calendar_estado",
            "google_event_id",
            "google_event_link",
            "observaciones",
        }
        pares = [(k, v) for k, v in cambios.items() if k in permitidos]
        if not pares:
            return self.obtener_agendamiento(negocio_id, id_agendamiento)
        set_sql = ", ".join(f"{k} = ?" for k, _ in pares)
        valores = [v for _, v in pares] + [negocio_id, id_agendamiento]
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE agendamientos SET {set_sql} WHERE negocio_id = ? AND id = ?",
                valores,
            )
            if cur.rowcount == 0:
                return None
            return self.obtener_agendamiento(negocio_id, id_agendamiento)

    def cancelar_agendamiento(self, negocio_id: str, id_agendamiento: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE agendamientos SET estado = 'cancelada' WHERE negocio_id = ? AND id = ?",
                (negocio_id, id_agendamiento),
            )
            return cur.rowcount > 0


class RepositorioComercialSupabase(RepositorioMultiTenant):
    """Implementacion multi-negocio sobre PostgREST HTTPS de Supabase."""

    modo_bd = "supabase_multitenant"

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        base = supabase_url.rstrip("/")
        self.base_rest = base if base.endswith("/rest/v1") else f"{base}/rest/v1"
        self._headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self._client = http_client or httpx.Client(timeout=15.0)

    def _req(self, metodo: str, ruta: str, **kwargs: Any) -> Any:
        headers = {**self._headers, **kwargs.pop("headers", {})}
        resp = self._client.request(metodo, f"{self.base_rest}/{ruta}", headers=headers, **kwargs)
        resp.raise_for_status()
        if not resp.text.strip():
            return []
        return resp.json()

    def listar_negocios(self) -> List[Dict[str, Any]]:
        filas = self._req("GET", "negocios?select=*&order=nombre_negocio.asc")
        return [_normalizar_negocio(f) for f in filas]

    def obtener_negocio(self, negocio_id: str) -> Optional[Dict[str, Any]]:
        filas = self._req("GET", f"negocios?id=eq.{quote(negocio_id, safe='')}&select=*&limit=1")
        return _normalizar_negocio(filas[0]) if filas else None

    def guardar_negocio(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        neg_id = normalizar_slug(str(datos.get("id") or datos.get("nombre_negocio") or ""))
        existente = self.obtener_negocio(neg_id)
        combinado = {**(existente or {}), **datos, "id": neg_id}
        if datos.get("pin_dueno"):
            combinado["pin_dueno_hash"] = hashear_pin(str(datos["pin_dueno"]))
        elif not combinado.get("pin_dueno_hash"):
            combinado["pin_dueno_hash"] = (existente or {}).get("pin_dueno_hash", "")
        if "whatsapp_access_token" not in datos and existente:
            combinado["whatsapp_access_token"] = existente.get("whatsapp_access_token", "")
        norm = _normalizar_negocio(combinado)
        self._validar_colision_canales(neg_id, norm)
        payload = {k: v for k, v in norm.items() if k != "tiene_pin_dueno"}
        try:
            res = self._req(
                "POST",
                "negocios?on_conflict=id",
                json=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise ConflictoCanalError(
                    "Ya existe otro negocio con ese canal de WhatsApp o Botpress."
                ) from exc
            raise
        return _normalizar_negocio(res[0]) if res else norm

    def eliminar_negocio(self, negocio_id: str) -> bool:
        res = self._req("DELETE", f"negocios?id=eq.{quote(negocio_id, safe='')}")
        return bool(res)

    def listar_servicios(self, negocio_id: str, solo_activos: bool = False) -> List[Dict[str, Any]]:
        q = f"servicios?negocio_id=eq.{quote(negocio_id, safe='')}&select=*&order=servicio_id_num.asc,nombre.asc"
        if solo_activos:
            q += "&activo=eq.true"
        try:
            filas = self._req("GET", q)
        except httpx.HTTPStatusError:
            filas = self._req(
                "GET",
                f"servicios?negocio_id=eq.{quote(negocio_id, safe='')}&select=*&order=servicio_id_num.asc",
            )
        salida = []
        for f in filas:
            f["palabras_clave"] = _parse_json_seguro(f.get("palabras_clave"), [])
            f["duracion_minutos"] = int(f.get("duracion_minutos") or 60)
            f["activo"] = bool(f.get("activo", True))
            f["tarifa_unitaria"] = float(f.get("tarifa_unitaria", 0.0))
            if solo_activos and not f["activo"]:
                continue
            salida.append(f)
        return salida

    def guardar_servicio(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]:
        clave = normalizar_slug(str(datos.get("clave") or datos.get("nombre") or ""))
        existentes = self.listar_servicios(negocio_id)
        previo = next((s for s in existentes if s["clave"] == clave), None)
        if datos.get("servicio_id_num"):
            num = int(datos["servicio_id_num"])
        elif previo:
            num = int(previo.get("servicio_id_num") or 1)
        else:
            num = max((int(s.get("servicio_id_num") or 0) for s in existentes), default=0) + 1
        palabras = datos.get("palabras_clave") or []
        if isinstance(palabras, str):
            palabras = [p.strip() for p in palabras.split(",") if p.strip()]
        payload = {
            "negocio_id": negocio_id,
            "clave": clave,
            "servicio_id_num": num,
            "nombre": str(datos.get("nombre") or clave),
            "descripcion": str(datos.get("descripcion") or ""),
            "tarifa_unitaria": float(datos.get("tarifa_unitaria", 0.0)),
            "palabras_clave": palabras,
            "requiere_complejidad": False,
            "costo_adicional_mantenimiento": 0.0,
            "duracion_minutos": max(15, int(datos.get("duracion_minutos") or 60)),
            "activo": bool(datos.get("activo", True)),
        }
        res = self._req(
            "POST",
            "servicios?on_conflict=negocio_id,clave",
            json=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        return res[0] if res else payload

    def eliminar_servicio(self, negocio_id: str, clave: str) -> bool:
        res = self._req(
            "DELETE",
            f"servicios?negocio_id=eq.{quote(negocio_id, safe='')}&clave=eq.{quote(clave, safe='')}",
        )
        return bool(res)

    def listar_promociones(self, negocio_id: str, solo_activas: bool = False) -> List[Dict[str, Any]]:
        q = f"promociones_vigentes?negocio_id=eq.{quote(negocio_id, safe='')}&select=*&order=codigo.asc"
        if solo_activas:
            q += "&activa=eq.true"
        filas = self._req("GET", q)
        for f in filas:
            f["activa"] = bool(f.get("activa", True))
            f["descuento_max"] = float(f.get("descuento_max", 0.0))
            f["condiciones"] = _parse_json_seguro(f.get("condiciones"), {})
        return filas

    def guardar_promocion(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]:
        codigo = normalizar_slug(str(datos.get("codigo") or "PROMO")).upper()
        payload = {
            "negocio_id": negocio_id,
            "codigo": codigo,
            "descripcion": str(datos.get("descripcion") or ""),
            "descuento_max": float(datos.get("descuento_max", 0.0)),
            "activa": bool(datos.get("activa", True)),
            "condiciones": _parse_json_seguro(datos.get("condiciones"), {}),
        }
        res = self._req(
            "POST",
            "promociones_vigentes?on_conflict=negocio_id,codigo",
            json=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        return res[0] if res else payload

    def eliminar_promocion(self, negocio_id: str, codigo: str) -> bool:
        res = self._req(
            "DELETE",
            f"promociones_vigentes?negocio_id=eq.{quote(negocio_id, safe='')}&codigo=eq.{quote(codigo, safe='')}",
        )
        return bool(res)

    def obtener_perfil(self, negocio_id: str, id_cliente: str) -> Optional[Dict[str, Any]]:
        filas = self._req(
            "GET",
            f"perfiles_clientes?negocio_id=eq.{quote(negocio_id, safe='')}&id_cliente=eq.{quote(id_cliente, safe='')}&select=*&limit=1",
        )
        if not filas:
            return None
        d = filas[0]
        d["historial_interacciones"] = _parse_json_seguro(d.get("historial_interacciones"), [])
        d["parametros_parciales"] = _parse_json_seguro(d.get("parametros_parciales"), {})
        d["automatizacion_activa"] = bool(d.get("automatizacion_activa", True))
        return d

    def obtener_o_crear_perfil(
        self, negocio_id: str, id_cliente: str, nombre: str = "Cliente", canal: str = "WhatsApp"
    ) -> Dict[str, Any]:
        existente = self.obtener_perfil(negocio_id, id_cliente)
        if existente:
            if (
                nombre
                and nombre not in _NOMBRES_PLACEHOLDER
                and existente.get("nombre", "") in _NOMBRES_PLACEHOLDER
            ):
                existente["nombre"] = nombre
                self.guardar_perfil(negocio_id, existente)
            return existente
        return self.guardar_perfil(
            negocio_id,
            {
                "id_cliente": id_cliente,
                "nombre": nombre or "Cliente",
                "canal": canal or "WhatsApp",
                "historial_interacciones": [],
                "tono_actual": "neutral",
                "fidelidad": 1,
                "ultima_cotizacion": None,
                "parametros_parciales": {},
                "automatizacion_activa": True,
            },
        )

    def guardar_perfil(self, negocio_id: str, perfil: Dict[str, Any]) -> Dict[str, Any]:
        ahora = datetime.now(timezone.utc).isoformat()
        payload = {
            "negocio_id": negocio_id,
            "id_cliente": str(perfil["id_cliente"]),
            "nombre": str(perfil.get("nombre") or "Cliente"),
            "canal": str(perfil.get("canal") or "WhatsApp"),
            "historial_interacciones": perfil.get("historial_interacciones", [])[-30:],
            "tono_actual": str(perfil.get("tono_actual") or "neutral"),
            "fidelidad": int(perfil.get("fidelidad", 1)),
            "ultima_cotizacion": perfil.get("ultima_cotizacion"),
            "parametros_parciales": perfil.get("parametros_parciales", {}),
            "automatizacion_activa": bool(perfil.get("automatizacion_activa", True)),
            "updated_at": ahora,
        }
        try:
            res = self._req(
                "POST",
                "perfiles_clientes?on_conflict=negocio_id,id_cliente",
                json=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            )
        except httpx.HTTPStatusError:
            payload.pop("updated_at", None)
            res = self._req(
                "POST",
                "perfiles_clientes?on_conflict=negocio_id,id_cliente",
                json=payload,
                headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            )
        return res[0] if res else payload

    def listar_perfiles(self, negocio_id: str, solo_pausados: bool = False) -> List[Dict[str, Any]]:
        q_base = f"perfiles_clientes?negocio_id=eq.{quote(negocio_id, safe='')}&select=*"
        if solo_pausados:
            q_base += "&automatizacion_activa=eq.false"
        try:
            filas = self._req("GET", q_base + "&order=updated_at.desc")
        except httpx.HTTPStatusError:
            filas = self._req("GET", q_base)
        for f in filas:
            f["historial_interacciones"] = _parse_json_seguro(f.get("historial_interacciones"), [])
            f["parametros_parciales"] = _parse_json_seguro(f.get("parametros_parciales"), {})
            f["automatizacion_activa"] = bool(f.get("automatizacion_activa", True))
        return filas

    def registrar_alerta(self, negocio_id: str, alerta: Dict[str, Any]) -> Dict[str, Any]:
        tz = self._tz_negocio(negocio_id)
        ts = str(alerta.get("timestamp") or datetime.now(tz).isoformat(timespec="seconds"))
        payload = {
            "negocio_id": negocio_id,
            "timestamp": ts,
            "id_cliente": str(alerta["id_cliente"]),
            "nombre": str(alerta.get("nombre") or "Cliente"),
            "canal": str(alerta.get("canal") or "WhatsApp"),
            "motivo_critico": str(alerta.get("motivo_critico") or ""),
            "resumen_ia": str(alerta.get("resumen_ia") or ""),
            "historial_charla": alerta.get("historial_charla", [])[-12:],
            "estado": str(alerta.get("estado") or "pausado"),
            "resuelta": False,
        }
        try:
            res = self._req("POST", "alertas_humanas", json=payload)
        except httpx.HTTPStatusError:
            payload.pop("resumen_ia", None)
            res = self._req("POST", "alertas_humanas", json=payload)
        return res[0] if res else payload

    def listar_alertas(self, negocio_id: str, solo_abiertas: bool = False) -> List[Dict[str, Any]]:
        q = f"alertas_humanas?negocio_id=eq.{quote(negocio_id, safe='')}&select=*&order=id.desc&limit=100"
        if solo_abiertas:
            q += "&resuelta=eq.false"
        filas = self._req("GET", q)
        for f in filas:
            f["historial_charla"] = _parse_json_seguro(f.get("historial_charla"), [])
            f["resuelta"] = bool(f.get("resuelta", False))
        return filas

    def resolver_alertas_cliente(self, negocio_id: str, id_cliente: str) -> None:
        self._req(
            "PATCH",
            f"alertas_humanas?negocio_id=eq.{quote(negocio_id, safe='')}&id_cliente=eq.{quote(id_cliente, safe='')}&resuelta=eq.false",
            json={"resuelta": True, "estado": "resuelta"},
        )

    def listar_agendamientos(
        self,
        negocio_id: str,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        solo_activos: bool = False,
    ) -> List[Dict[str, Any]]:
        q = f"agendamientos?negocio_id=eq.{quote(negocio_id, safe='')}&select=*&order=fecha_inicio.asc"
        if solo_activos:
            q += "&estado=in.(pendiente,confirmada)"
        if fecha_inicio:
            q += f"&fecha_fin=gt.{quote(fecha_inicio, safe='')}"
        if fecha_fin:
            q += f"&fecha_inicio=lt.{quote(fecha_fin, safe='')}"
        filas = self._req("GET", q)
        for f in filas:
            f["metadata"] = _parse_json_seguro(f.get("metadata"), {})
        return filas

    def obtener_agendamiento(self, negocio_id: str, id_agendamiento: str) -> Optional[Dict[str, Any]]:
        if not _es_uuid_valido(id_agendamiento):
            return None
        filas = self._req(
            "GET",
            f"agendamientos?negocio_id=eq.{quote(negocio_id, safe='')}&id=eq.{quote(id_agendamiento, safe='')}&select=*&limit=1",
        )
        if not filas:
            return None
        d = filas[0]
        d["metadata"] = _parse_json_seguro(d.get("metadata"), {})
        return d

    def registrar_agendamiento(self, negocio_id: str, datos: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "id": str(datos.get("id") or uuid.uuid4()),
            "negocio_id": negocio_id,
            "id_cliente": str(datos.get("id_cliente") or ""),
            "nombre_cliente": str(datos.get("nombre_cliente") or "Cliente"),
            "canal": str(datos.get("canal") or "WhatsApp"),
            "servicio": str(datos.get("servicio") or ""),
            "fecha_inicio": str(datos["fecha_inicio"]),
            "fecha_fin": str(datos["fecha_fin"]),
            "estado": str(datos.get("estado") or "confirmada"),
            "registrado_en": str(datos.get("registrado_en") or "solo_supabase"),
            "google_calendar_estado": str(datos.get("google_calendar_estado") or "no_configurado"),
            "google_event_id": datos.get("google_event_id"),
            "google_event_link": datos.get("google_event_link"),
            "observaciones": str(datos.get("observaciones") or ""),
            "metadata": datos.get("metadata") or {},
        }
        try:
            res = self._req("POST", "agendamientos", json=payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409 or "23P01" in exc.response.text:
                raise ConflictoAgendaError(
                    "El horario solicitado se cruza con otra cita activa."
                ) from exc
            raise
        return res[0] if res else payload

    def actualizar_agendamiento(
        self, negocio_id: str, id_agendamiento: str, cambios: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not _es_uuid_valido(id_agendamiento):
            return None
        permitidos = {
            "estado",
            "registrado_en",
            "google_calendar_estado",
            "google_event_id",
            "google_event_link",
            "observaciones",
        }
        payload = {k: v for k, v in cambios.items() if k in permitidos}
        if not payload:
            return self.obtener_agendamiento(negocio_id, id_agendamiento)
        res = self._req(
            "PATCH",
            f"agendamientos?negocio_id=eq.{quote(negocio_id, safe='')}&id=eq.{quote(id_agendamiento, safe='')}",
            json=payload,
        )
        return res[0] if res else None

    def cancelar_agendamiento(self, negocio_id: str, id_agendamiento: str) -> bool:
        if not _es_uuid_valido(id_agendamiento):
            return False
        res = self._req(
            "PATCH",
            f"agendamientos?negocio_id=eq.{quote(negocio_id, safe='')}&id=eq.{quote(id_agendamiento, safe='')}",
            json={"estado": "cancelada"},
        )
        return bool(res)
