"""Comprueba la conexion FreeBusy de Google Calendar sin modificar eventos."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import google.auth.exceptions
import httpx

from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.persistencia.google_calendar import ErrorGoogleCalendar
from src.persistencia.repositorio import RepositorioSQLite

_GOOGLE_CALENDAR_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/calendar.readonly"
)


def _siguiente_intervalo(
    zona: ZoneInfo,
) -> Tuple[datetime, datetime]:
    candidato = (datetime.now(zona) + timedelta(days=1)).replace(
        second=0, microsecond=0
    )
    minutos_hasta_bloque = 30 - candidato.minute % 30
    inicio = candidato + timedelta(minutes=minutos_hasta_bloque)
    return inicio, inicio + timedelta(minutes=30)


def _clasificar_error(error: ErrorGoogleCalendar) -> Tuple[str, str]:
    causa = error.__cause__
    if isinstance(causa, httpx.HTTPStatusError):
        estado = causa.response.status_code
        if estado == 401:
            categoria = "credenciales o autenticacion"
        elif estado == 403:
            categoria = "permisos de la cuenta de servicio sobre el calendario"
        elif estado == 404:
            categoria = "ID o acceso al calendario"
        elif estado == 400:
            categoria = "configuracion o intervalo de la solicitud"
        elif estado == 429 or estado >= 500:
            categoria = "disponibilidad de la API de Google"
        else:
            categoria = "respuesta HTTP de Google Calendar"
        return categoria, f"HTTP {estado}"

    if isinstance(causa, FileNotFoundError):
        return "ruta del archivo de credenciales", "no se encontro el archivo configurado"
    if isinstance(causa, google.auth.exceptions.GoogleAuthError):
        return "credenciales o autenticacion", type(causa).__name__
    if isinstance(causa, httpx.TimeoutException):
        return "red o disponibilidad de la API", "tiempo de espera agotado"
    if isinstance(causa, httpx.RequestError):
        return "red o disponibilidad de la API", type(causa).__name__
    if isinstance(causa, ValueError):
        return (
            "formato de credenciales o respuesta JSON de la API",
            type(causa).__name__,
        )

    detalle = str(error)
    texto = detalle.lower()
    if "forbidden" in texto:
        return "permisos de la cuenta de servicio sobre el calendario", detalle
    if "notfound" in texto or "notacalendar" in texto:
        return "ID o tipo de calendario", detalle
    if "internalerror" in texto or "backenderror" in texto:
        return "disponibilidad de la API de Google", detalle
    if "calendario configurado" in texto:
        return "ID, permisos o respuesta del calendario", detalle
    if "respuesta de disponibilidad" in texto or "intervalos de disponibilidad" in texto:
        return "formato de respuesta de FreeBusy", detalle
    return "Google Calendar", detalle


def main() -> int:
    raiz = Path(__file__).resolve().parent
    try:
        config = ConfiguracionBotpress.desde_entorno(raiz / ".env")
    except (OSError, ValueError) as exc:
        print(f"ERROR de configuracion ({type(exc).__name__}); no se muestran valores.")
        return 2

    if not config.usa_google_calendar:
        print(
            "ERROR de configuracion: se requieren GOOGLE_CALENDAR_ID y "
            "GOOGLE_SERVICE_ACCOUNT_JSON. No se muestran sus valores."
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
    except (OSError, ValueError, ZoneInfoNotFoundError) as exc:
        print(
            "ERROR de configuracion o inicializacion "
            f"({type(exc).__name__}); no se muestran valores."
        )
        return 2

    calendario = servicio.google_calendar
    if calendario is None:
        print("ERROR de configuracion: el servicio no inicializo Google Calendar.")
        return 2
    if servicio.entorno.agendaVirtual.google_calendar is not calendario:
        print("ERROR de inicializacion: Google Calendar no se inyecto en el entorno.")
        return 3
    if servicio.agendamientos.entorno is not servicio.entorno:
        print("ERROR de inicializacion: el agente no comparte el entorno del servicio.")
        return 3

    try:
        credenciales = calendario._cargar_credenciales()
        if credenciales is None:
            print("ERROR de credenciales: no se pudo cargar la cuenta de servicio.")
            return 4
        calendario._credenciales = credenciales.with_scopes(
            [_GOOGLE_CALENDAR_READONLY_SCOPE]
        )
    except FileNotFoundError:
        print("ERROR de credenciales: no se encontro el archivo configurado.")
        return 4
    except google.auth.exceptions.GoogleAuthError as exc:
        print(f"ERROR de credenciales ({type(exc).__name__}); no se muestran valores.")
        return 4
    except (OSError, ValueError) as exc:
        print(
            f"ERROR al leer las credenciales ({type(exc).__name__}); "
            "no se muestran valores."
        )
        return 4

    try:
        zona = ZoneInfo(config.google_timezone)
        inicio, fin = _siguiente_intervalo(zona)
        inicio_parseado, _ = calendario._parsear_bloque(inicio.isoformat())
        if inicio_parseado != inicio:
            print("ERROR: el cliente no interpreto el horario en la zona configurada.")
            return 4
        ocupados = calendario._consultar_freebusy(inicio, fin)
        if any(
            desde.tzinfo is None
            or hasta.tzinfo is None
            or desde.utcoffset() != inicio.utcoffset()
            or hasta.utcoffset() != inicio.utcoffset()
            or hasta <= desde
            for desde, hasta in ocupados
        ):
            print("ERROR: FreeBusy devolvio horarios invalidos o sin zona horaria.")
            return 4
    except ZoneInfoNotFoundError:
        print("ERROR de configuracion: GOOGLE_TIMEZONE no es una zona IANA valida.")
        return 2
    except ErrorGoogleCalendar as exc:
        categoria, detalle = _clasificar_error(exc)
        print(f"ERROR ({categoria}): {detalle}")
        return 4

    ocupado = any(inicio < hasta and fin > desde for desde, hasta in ocupados)
    estado = "OCUPADO" if ocupado else "LIBRE"
    print("Google Calendar FreeBusy: respuesta valida.")
    print(f"Calendario: configurado (ID omitido); zona: {config.google_timezone}.")
    print(f"Intervalo consultado: {inicio.isoformat()} a {fin.isoformat()} (30 minutos).")
    print(f"Disponibilidad: {estado}.")
    print("Operacion exclusivamente de lectura; no se modificaron eventos ni Supabase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
