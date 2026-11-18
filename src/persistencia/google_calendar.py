"""
Conector oficial con Google Calendar API v3 mediante Service Account (google-auth)
y peticiones REST HTTPS con httpx.

Permite a EntornoVirtualAtencion y AgenteAgendamientos consultar disponibilidad
real (freeBusy), proponer huecos libres del dia y agendar citas reales en el
calendario del dueno del negocio.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import google.auth.transport.requests
import httpx
from google.oauth2 import service_account

_SCOPES_CALENDAR = ["https://www.googleapis.com/auth/calendar"]
_BASE_CALENDAR_API = "https://www.googleapis.com/calendar/v3"


class ClienteGoogleCalendar:
    """Cliente ligero para Google Calendar API v3 usando Service Account."""

    def __init__(
        self,
        calendar_id: str,
        service_account_json: str = "",
        timezone: str = "America/Bogota",
        duracion_minutos: int = 60,
        hora_inicio_laboral: int = 8,
        hora_fin_laboral: int = 18,
        http_client: Optional[httpx.Client] = None,
        token_estatico: Optional[str] = None,
    ) -> None:
        self.calendar_id = calendar_id.strip()
        self.service_account_json = service_account_json.strip()
        self.timezone = timezone.strip() or "America/Bogota"
        self.tz = ZoneInfo(self.timezone)
        self.duracion_minutos = duracion_minutos
        self.hora_inicio_laboral = hora_inicio_laboral
        self.hora_fin_laboral = hora_fin_laboral
        self._client = http_client or httpx.Client(trust_env=False, timeout=12.0)
        self._token_estatico = token_estatico
        self._credenciales = self._cargar_credenciales()

    def _cargar_credenciales(self) -> Optional[service_account.Credentials]:
        if self._token_estatico or not self.service_account_json:
            return None
        valor = self.service_account_json
        if valor.startswith("{"):
            info = json.loads(valor)
            return service_account.Credentials.from_service_account_info(
                info, scopes=_SCOPES_CALENDAR
            )
        ruta = Path(valor)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo de Service Account de Google: '{ruta}'"
            )
        return service_account.Credentials.from_service_account_file(
            str(ruta), scopes=_SCOPES_CALENDAR
        )

    def _headers(self) -> Dict[str, str]:
        if self._token_estatico:
            token = self._token_estatico
        else:
            if self._credenciales is None:
                raise RuntimeError(
                    "Faltan credenciales GOOGLE_SERVICE_ACCOUNT_JSON para Google Calendar."
                )
            if not self._credenciales.valid:
                self._credenciales.refresh(google.auth.transport.requests.Request())
            token = str(self._credenciales.token)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _parsear_bloque(self, fecha_hora: str) -> Tuple[datetime, datetime]:
        dt_naive = datetime.strptime(fecha_hora.strip(), "%Y-%m-%d %H:%M")
        inicio = dt_naive.replace(tzinfo=self.tz)
        fin = inicio + timedelta(minutes=self.duracion_minutos)
        return inicio, fin

    def _consultar_freebusy(
        self, time_min: datetime, time_max: datetime
    ) -> List[Tuple[datetime, datetime]]:
        resp = self._client.post(
            f"{_BASE_CALENDAR_API}/freeBusy",
            headers=self._headers(),
            json={
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "timeZone": self.timezone,
                "items": [{"id": self.calendar_id}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        cal_info = data.get("calendars", {}).get(self.calendar_id, {})
        ocupados: List[Tuple[datetime, datetime]] = []
        for tramo in cal_info.get("busy", []):
            ini = datetime.fromisoformat(tramo["start"].replace("Z", "+00:00")).astimezone(self.tz)
            fin = datetime.fromisoformat(tramo["end"].replace("Z", "+00:00")).astimezone(self.tz)
            ocupados.append((ini, fin))
        return ocupados

    def verificar_disponibilidad(self, fecha_hora: str) -> bool:
        """Retorna True si el bloque solicitado esta libre en Google Calendar."""
        inicio, fin = self._parsear_bloque(fecha_hora)
        ocupados = self._consultar_freebusy(inicio, fin)
        return len(ocupados) == 0

    def obtener_alternativas(self, excluir_fecha: str, limite: int = 2) -> List[str]:
        """Consulta bloques libres reales del mismo dia dentro de horario laboral."""
        inicio_ref, _ = self._parsear_bloque(excluir_fecha)
        jornada_ini = inicio_ref.replace(
            hour=self.hora_inicio_laboral, minute=0, second=0, microsecond=0
        )
        jornada_fin = inicio_ref.replace(
            hour=self.hora_fin_laboral, minute=0, second=0, microsecond=0
        )
        ocupados = self._consultar_freebusy(jornada_ini, jornada_fin)

        alternativas: List[str] = []
        cursor = jornada_ini
        paso = timedelta(minutes=self.duracion_minutos)
        while cursor + paso <= jornada_fin and len(alternativas) < limite:
            slot_fin = cursor + paso
            etiqueta = cursor.strftime("%Y-%m-%d %H:%M")
            se_cruza = any(cursor < b_fin and slot_fin > b_ini for b_ini, b_fin in ocupados)
            if not se_cruza and etiqueta != excluir_fecha.strip():
                alternativas.append(etiqueta)
            cursor = slot_fin
        return alternativas

    def crear_evento(
        self, fecha_hora: str, cliente: str, descripcion: str = ""
    ) -> Dict[str, Any]:
        """Crea una cita confirmada real en el Google Calendar configurado."""
        inicio, fin = self._parsear_bloque(fecha_hora)
        cal_id_enc = quote(self.calendar_id, safe="")
        resp = self._client.post(
            f"{_BASE_CALENDAR_API}/calendars/{cal_id_enc}/events",
            headers=self._headers(),
            json={
                "summary": f"Cita Comercial - {cliente}",
                "description": descripcion
                or f"Reserva confirmada desde SistemaMultiagenteComercial para {cliente}.",
                "start": {"dateTime": inicio.isoformat(), "timeZone": self.timezone},
                "end": {"dateTime": fin.isoformat(), "timeZone": self.timezone},
            },
        )
        resp.raise_for_status()
        evento = resp.json()
        return {
            "fecha_hora": fecha_hora,
            "estado": "OCUPADO",
            "cliente": cliente,
            "evento_google_id": evento.get("id"),
            "html_link": evento.get("htmlLink"),
        }

    def listar_proximos_eventos(self, limite: int = 8) -> Dict[str, Dict[str, Any]]:
        """Obtiene eventos reales futuros para mostrarlos en el monitor web."""
        ahora = datetime.now(self.tz)
        cal_id_enc = quote(self.calendar_id, safe="")
        resp = self._client.get(
            f"{_BASE_CALENDAR_API}/calendars/{cal_id_enc}/events",
            headers=self._headers(),
            params={
                "timeMin": ahora.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": str(limite),
            },
        )
        resp.raise_for_status()
        bloques: Dict[str, Dict[str, Any]] = {}
        for item in resp.json().get("items", []):
            start_raw = item.get("start", {}).get("dateTime") or item.get("start", {}).get("date")
            if not start_raw:
                continue
            try:
                dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(self.tz)
                clave = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                clave = str(start_raw)
            bloques[clave] = {
                "estado": "OCUPADO",
                "cliente": item.get("summary", "Evento Google Calendar"),
            }
        return bloques
