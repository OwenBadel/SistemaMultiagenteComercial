"""Reglas verificables para promociones y datos de reserva."""
from __future__ import annotations

from datetime import date, datetime
from math import isfinite
import re
from typing import Any, Dict
import unicodedata
from zoneinfo import ZoneInfo


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
