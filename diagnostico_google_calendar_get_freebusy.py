"""Diagnostico de solo lectura para calendars.get y FreeBusy."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import google.auth.exceptions
import httpx

from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.persistencia.google_calendar import ErrorGoogleCalendar
from src.persistencia.repositorio import RepositorioSQLite

_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
_CALENDAR_API = "https://www.googleapis.com/calendar/v3"


def _razones_seguras(error: Any) -> List[str]:
    if not isinstance(error, dict):
        return []
    errores = error.get("errors", [])
    if not isinstance(errores, list):
        return []
    razones = {
        item["reason"]
        for item in errores
        if isinstance(item, dict)
        and isinstance(item.get("reason"), str)
        and item["reason"].isascii()
        and item["reason"].replace("_", "").replace("-", "").isalnum()
        and len(item["reason"]) <= 64
    }
    return sorted(razones)


def _estado_http(respuesta: httpx.Response) -> str:
    try:
        cuerpo = respuesta.json()
    except ValueError:
        cuerpo = {}
    error = cuerpo.get("error", {}) if isinstance(cuerpo, dict) else {}
    if not isinstance(error, dict):
        error = {}
    status_texto = error.get("status")
    status_texto_seguro = (
        status_texto
        if isinstance(status_texto, str)
        and status_texto.isascii()
        and status_texto.replace("_", "").isalnum()
        and len(status_texto) <= 64
        else None
    )
    razones = _razones_seguras(error)
    detalle = [f"HTTP {respuesta.status_code}"]
    if status_texto_seguro:
        detalle.append(f"status={status_texto_seguro}")
    if razones:
        detalle.append(f"reason={','.join(razones)}")
    return " ".join(detalle)


def _siguiente_intervalo(zona: ZoneInfo) -> Tuple[datetime, datetime]:
    candidato = (datetime.now(zona) + timedelta(days=1)).replace(
        second=0, microsecond=0
    )
    inicio = candidato + timedelta(minutes=30 - candidato.minute % 30)
    return inicio, inicio + timedelta(minutes=30)


def _consultar_calendario(
    servicio: ServicioMultiagenteComercial,
) -> Tuple[Optional[httpx.Response], Optional[str]]:
    calendario = servicio.google_calendar
    if calendario is None:
        return None, "Google Calendar no se inicializo."
    calendar_id_enc = quote(calendario.calendar_id, safe="")
    try:
        respuesta = calendario._client.get(
            f"{_CALENDAR_API}/calendars/{calendar_id_enc}",
            headers=calendario._headers(),
        )
        return respuesta, None
    except httpx.RequestError as exc:
        return None, f"error de transporte ({type(exc).__name__})"
    except google.auth.exceptions.GoogleAuthError as exc:
        return None, f"error de autenticacion ({type(exc).__name__})"
    except (OSError, ValueError) as exc:
        return None, f"error al leer credenciales ({type(exc).__name__})"


def _consultar_freebusy(
    servicio: ServicioMultiagenteComercial,
    inicio: datetime,
    fin: datetime,
) -> Tuple[Optional[List[Tuple[datetime, datetime]]], Optional[str]]:
    calendario = servicio.google_calendar
    if calendario is None:
        return None, "Google Calendar no se inicializo."
    try:
        intervalos = calendario._consultar_freebusy(inicio, fin)
        return intervalos, None
    except ErrorGoogleCalendar as exc:
        causa = exc.__cause__
        if isinstance(causa, httpx.HTTPStatusError):
            return None, _estado_http(causa.response)
        return None, str(exc)


def main() -> int:
    raiz = Path(__file__).resolve().parent
    try:
        config = ConfiguracionBotpress.desde_entorno(raiz / ".env")
    except (OSError, ValueError) as exc:
        print(f"Configuracion: error ({type(exc).__name__}); valores omitidos.")
        return 2
    if not config.usa_google_calendar:
        print(
            "Configuracion: faltan GOOGLE_CALENDAR_ID o "
            "GOOGLE_SERVICE_ACCOUNT_JSON; valores omitidos."
        )
        return 2

    repositorio = RepositorioSQLite(
        ruta_db=":memory:", negocio_id=config.negocio_id
    )
    try:
        servicio = ServicioMultiagenteComercial(
            config_botpress=config,
            repositorio=repositorio,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
    except (OSError, ValueError) as exc:
        print(f"Inicializacion: error ({type(exc).__name__}); valores omitidos.")
        return 2

    calendario = servicio.google_calendar
    if calendario is None:
        print("Inicializacion: Google Calendar no se creo.")
        return 2
    if servicio.entorno.agendaVirtual.google_calendar is not calendario:
        print("Inicializacion: cliente de calendario no inyectado en el entorno.")
        return 2

    try:
        credenciales = calendario._cargar_credenciales()
        if credenciales is None:
            print("Credenciales: no se pudo cargar la cuenta de servicio.")
            return 2
        calendario._credenciales = credenciales.with_scopes([_READONLY_SCOPE])
    except FileNotFoundError:
        print("Credenciales: no se encontro el archivo configurado.")
        return 2
    except google.auth.exceptions.GoogleAuthError as exc:
        print(f"Credenciales: error ({type(exc).__name__}); valores omitidos.")
        return 2
    except (OSError, ValueError) as exc:
        print(f"Credenciales: error de lectura ({type(exc).__name__}); valores omitidos.")
        return 2

    print(f"GOOGLE_CALENDAR_ID efectivo: {calendario.calendar_id}")
    print(f"Scope de esta prueba: {_READONLY_SCOPE}")

    respuesta_get, error_get = _consultar_calendario(servicio)
    if respuesta_get is None:
        print(f"calendars.get: {error_get}")
    elif respuesta_get.is_success:
        try:
            datos_get = respuesta_get.json()
        except ValueError:
            datos_get = None
        id_coincide = (
            isinstance(datos_get, dict)
            and datos_get.get("id") == calendario.calendar_id
        )
        print(
            f"calendars.get: {_estado_http(respuesta_get)}; "
            f"returned_id_matches={id_coincide}"
        )
    else:
        print(f"calendars.get: {_estado_http(respuesta_get)}")

    zona = ZoneInfo(config.google_timezone)
    inicio, fin = _siguiente_intervalo(zona)
    intervalos, error_freebusy = _consultar_freebusy(servicio, inicio, fin)
    if intervalos is None:
        print(f"FreeBusy: {error_freebusy}")
    else:
        intervalos_validos = all(
            desde.tzinfo is not None
            and hasta.tzinfo is not None
            and hasta > desde
            for desde, hasta in intervalos
        )
        ocupado = any(inicio < hasta and fin > desde for desde, hasta in intervalos)
        print(
            f"FreeBusy: HTTP 200; intervalos_validos={intervalos_validos}; "
            f"disponibilidad={'OCUPADO' if ocupado else 'LIBRE'}"
        )
    print(f"Intervalo FreeBusy: {inicio.isoformat()} a {fin.isoformat()}")
    print("Solo lectura: no se crean, modifican ni eliminan eventos; Supabase no se usa.")
    return 0 if respuesta_get is not None and respuesta_get.is_success and intervalos is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
