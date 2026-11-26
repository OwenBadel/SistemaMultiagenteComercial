"""Reglas verificables para promociones y datos de reserva."""
from __future__ import annotations

from datetime import date, datetime
from math import ceil, isfinite
import re
from typing import Any, Dict, Optional
import unicodedata
from zoneinfo import ZoneInfo


ENTREGABLES_BASE_PAQUETES: Dict[str, int] = {
    "sesion_producto": 10,
    "retrato_corporativo": 8,
    "cobertura_evento": 30,
    "video_reel": 2,
}


def _redondear_5000(valor: float) -> float:
    """Redondea al multiplo de $5,000 COP mas cercano (hacia arriba si es piso)."""
    if valor <= 0:
        return 0.0
    return float(round(valor / 5000.0) * 5000)


def _techo_5000(valor: float) -> float:
    """Redondea hacia arriba al siguiente multiplo de $5,000 COP para pisos de rentabilidad."""
    if valor <= 0:
        return 0.0
    return float(ceil(valor / 5000.0) * 5000)


def extraer_entregables_de_descripcion(
    clave_servicio: str, descripcion: str = ""
) -> int:
    """Obtiene cuantas fotos o piezas incluye un paquete desde su descripcion en BD o default."""
    match = re.search(r"(\d+)\s*(?:fotos?|fotograf[ií]as?|reels?|videos?)", descripcion or "", flags=re.IGNORECASE)
    if match:
        valor = int(match.group(1))
        if valor > 0:
            return valor
    return ENTREGABLES_BASE_PAQUETES.get(clave_servicio, 8)


def calcular_cotizacion_a_medida_fotografia(
    *,
    clave_servicio_ref: str,
    nombre_paquete_ref: str,
    tarifa_paquete_ref: float,
    descripcion_paquete_ref: str = "",
    descripcion_cliente: str = "Servicio Fotografico Personalizado",
    margen_base: float = 0.40,
    margen_minimo: float = 0.25,
    cantidad_fotos: Optional[int] = None,
    cantidad_personas: Optional[int] = None,
    en_exterior: bool = False,
    cantidad_sesiones: int = 1,
    urgencia_dias: Optional[int] = None,
    moneda: str = "COP",
) -> Dict[str, Any]:
    """
    Calcula una cotizacion a medida o fraccionada (ej. 1 sola foto o sesion familiar)
    derivando economia unitaria desde el paquete de referencia en BD y garantizando
    un piso minimo anti-perdida (margen_neto >= margen_minimo).
    """
    margen_obj = min(max(float(margen_base or 0.40), 0.25), 0.70)
    margen_min = min(max(float(margen_minimo or 0.25), 0.15), margen_obj)
    sesiones = max(1, int(cantidad_sesiones or 1))
    tarifa_ref = max(100_000.0, float(tarifa_paquete_ref or 300_000.0))

    fotos_paquete = extraer_entregables_de_descripcion(
        clave_servicio_ref, descripcion_paquete_ref
    )
    fotos_solicitadas = (
        max(1, int(cantidad_fotos))
        if cantidad_fotos is not None and int(cantidad_fotos) > 0
        else fotos_paquete
    )
    personas = (
        max(1, int(cantidad_personas))
        if cantidad_personas is not None and int(cantidad_personas) > 0
        else 1
    )

    # Descomposicion de economia unitaria derivada del paquete real en BD
    costo_paquete_ref = tarifa_ref * (1.0 - margen_obj)
    costo_alistamiento = max(55_000.0, round((costo_paquete_ref * 0.30) / 1000.0) * 1000.0)
    costo_variable_paquete = max(0.0, costo_paquete_ref - costo_alistamiento)
    costo_por_foto = max(
        10_000.0,
        min(25_000.0, round((costo_variable_paquete / max(1, fotos_paquete)) / 1000.0) * 1000.0),
    )

    personas_extra = max(0, personas - 2)
    costo_personas_extra = personas_extra * 15_000.0
    costo_desplazamiento = 55_000.0 if en_exterior else 0.0

    costo_operativo_sesion = (
        costo_alistamiento
        + (fotos_solicitadas * costo_por_foto)
        + costo_personas_extra
        + costo_desplazamiento
    )
    if urgencia_dias is not None and int(urgencia_dias) <= 2:
        costo_operativo_sesion *= 1.20

    costo_operativo_total = round(costo_operativo_sesion * sesiones, 2)
    piso_minimo_rentable = _techo_5000(costo_operativo_total / (1.0 - margen_min))
    precio_objetivo = _redondear_5000(costo_operativo_total / (1.0 - margen_obj))

    # Coherencia comercial: si pide <= fotos del paquete base en estudio sin extras ni urgencia,
    # nunca debe costar mas que el paquete completo de referencia.
    aplica_tope_paquete = (
        not en_exterior
        and personas_extra == 0
        and (urgencia_dias is None or int(urgencia_dias) > 2)
        and fotos_solicitadas <= fotos_paquete
    )
    conveniencia_paquete = False
    if aplica_tope_paquete and precio_objetivo >= tarifa_ref * sesiones:
        precio_objetivo = float(tarifa_ref * sesiones)
        conveniencia_paquete = True

    valor_total = max(precio_objetivo, piso_minimo_rentable)
    margen_neto_real = (
        round((valor_total - costo_operativo_total) / valor_total, 4)
        if valor_total > 0
        else 0.0
    )

    return {
        "valor_total": valor_total,
        "desglose": {
            "servicio": descripcion_cliente,
            "es_personalizado": True,
            "paquete_referencia": nombre_paquete_ref,
            "clave_referencia": clave_servicio_ref,
            "cantidad_unidades": sesiones,
            "unidad_cantidad": "sesiones",
            "cantidad_fotos": fotos_solicitadas,
            "fotos_paquete_referencia": fotos_paquete,
            "cantidad_personas": personas,
            "locacion": "exterior" if en_exterior else "estudio",
            "tarifa_unitaria": round(valor_total / sesiones, 2),
            "costo_base": round(costo_alistamiento * sesiones, 2),
            "costo_operativo_directo": costo_operativo_total,
            "piso_minimo_rentable": piso_minimo_rentable,
            "margen_minimo_requerido": margen_min,
            "margen_neto_estimado": margen_neto_real,
            "ajuste_complejidad_urgencia": round(
                (costo_personas_extra + costo_desplazamiento) * sesiones, 2
            ),
            "adicional_mantenimiento": 0.0,
            "conveniencia_paquete_completo": conveniencia_paquete,
            "moneda": moneda,
        },
    }


def promocion_elegible(
    promocion: Dict[str, Any],
    servicio: str,
    cantidad: int,
    complejidad: int | None = None,
    hoy: date | None = None,
    canal: str | None = None,
) -> bool:
    """Falla cerrada si una promoción no tiene condiciones estructuradas."""
    if promocion.get("activa") is not True:
        return False
    try:
        descuento_maximo = float(promocion.get("descuento_max", 0))
    except (TypeError, ValueError):
        return False
    if not isfinite(descuento_maximo) or descuento_maximo <= 0:
        return False

    condiciones = promocion.get("condiciones")
    if not isinstance(condiciones, dict):
        return False
    claves_admitidas = {
        "servicios",
        "servicio",
        "cantidad_minima",
        "cantidad_maxima",
        "complejidad_minima",
        "complejidad_maxima",
        "fecha_inicio",
        "fecha_fin",
        "canales",
    }
    if set(condiciones) - claves_admitidas:
        return False

    servicio = str(servicio or "").strip()
    try:
        cantidad = int(cantidad)
    except (TypeError, ValueError):
        return False
    if not servicio or cantidad <= 0:
        return False

    servicios = condiciones.get("servicios")
    if servicios is None and condiciones.get("servicio") is not None:
        servicios = [condiciones["servicio"]]
    if not isinstance(servicios, (list, tuple, set)) or servicio not in {
        str(item) for item in servicios
    }:
        return False
    canales = condiciones.get("canales")
    if canales is not None and (
        not canal
        or not isinstance(canales, (list, tuple, set))
        or str(canal) not in {str(item) for item in canales}
    ):
        return False

    try:
        minimo = int(condiciones.get("cantidad_minima", 1))
        maximo = int(condiciones.get("cantidad_maxima", 2**31 - 1))
    except (TypeError, ValueError):
        return False
    if not minimo <= cantidad <= maximo:
        return False

    if "complejidad_minima" in condiciones or "complejidad_maxima" in condiciones:
        if complejidad is None:
            return False
        try:
            complejidad_minima = int(condiciones.get("complejidad_minima", 0))
            complejidad_maxima = int(condiciones.get("complejidad_maxima", 2**31 - 1))
        except (TypeError, ValueError):
            return False
        if not complejidad_minima <= complejidad <= complejidad_maxima:
            return False

    fecha_actual = hoy or datetime.now(ZoneInfo("America/Bogota")).date()
    desde = condiciones.get("fecha_inicio")
    hasta = condiciones.get("fecha_fin")
    try:
        if desde and fecha_actual < date.fromisoformat(str(desde)):
            return False
        if hasta and fecha_actual > date.fromisoformat(str(hasta)):
            return False
    except ValueError:
        return False
    return True


def validar_datos_cliente_reserva(
    nombre: Any, identificacion: Any, telefono: Any
) -> Dict[str, str]:
    nombre_normalizado = " ".join(str(nombre or "").split())
    nombre_busqueda = unicodedata.normalize("NFD", nombre_normalizado.casefold())
    nombre_busqueda = "".join(
        caracter
        for caracter in nombre_busqueda
        if unicodedata.category(caracter) != "Mn"
    )
    tokens_nombre = re.findall(r"[^\W\d_]+", nombre_normalizado, flags=re.UNICODE)
    if (
        len(tokens_nombre) < 2
        or nombre_busqueda in {
            "cliente",
            "cliente potencial",
            "cliente empresarial",
            "anonymous user",
        }
    ):
        raise ValueError("Se requiere el nombre completo del cliente.")

    identificacion_normalizada = re.sub(r"\D", "", str(identificacion or ""))
    if not 5 <= len(identificacion_normalizada) <= 15:
        raise ValueError("La identificación debe contener entre 5 y 15 dígitos.")

    telefono_texto = str(telefono or "").strip()
    telefono_normalizado = re.sub(r"\D", "", telefono_texto)
    if not 7 <= len(telefono_normalizado) <= 15:
        raise ValueError("El teléfono debe contener entre 7 y 15 dígitos.")
    if telefono_texto.startswith("+"):
        telefono_normalizado = f"+{telefono_normalizado}"
    return {
        "nombre_cliente": nombre_normalizado,
        "cedula": identificacion_normalizada,
        "telefono": telefono_normalizado,
    }
