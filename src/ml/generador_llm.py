"""
Generador de lenguaje natural (NLG) basado en Gemini REST API con respaldo
determinista automatico (fallback) y circuit breaker anti-bloqueo.

Rol en el SistemaMultiagenteComercial:
- Scikit-Learn (ModeloNLPVentas, RandomForest, KNN) ejecuta la clasificacion,
  extraccion de entidades, calculos economicos y reglas deterministas (< 1 ms).
- GeneradorDialogoGemini redacta la respuesta saliente de AgenteVentas con
  tono conversacional natural, preservando estrictamente los hechos verificados
  (montos, porcentajes, fechas) y cayendo al borrador local sin bloquearse si
  se agotan los tokens (HTTP 429), falla la clave (401/403) o hay timeout.
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

_URL_BASE_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
_PATRON_MONTO = re.compile(r"\$\s*[\d.,]+")

_INSTRUCCION_SISTEMA = (
    "Eres el asesor comercial de un negocio profesional atendiendo clientes por chat. "
    "Redacta respuestas claras, cercanas, naturales, variadas y concisas en espanol "
    "(maximo 4 oraciones o lista corta si muestras un catalogo). "
    "Reglas estrictas e inviolables:\n"
    "1. Nunca inventes precios, descuentos, porcentajes, servicios ni fechas/horarios "
    "que no figuren en los datos verificados.\n"
    "2. Conserva de forma exacta todos los montos en dinero ($...), porcentajes (%) "
    "y fechas/horas suministrados.\n"
    "3. Nunca menciones tecnicismos internos como 'complejidad 1-5', 'parametros obligatorios', "
    "'FIPA', 'SPADE', 'Random Forest', 'KNN', 'base de datos' ni nombres de variables.\n"
    "4. No uses emojis.\n"
    "5. Responde directamente al cliente sin preambulos ni comillas envolventes."
)


def _normalizar_comparacion(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.casefold())
    sin_tildes = "".join(
        c for c in descompuesto if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_tildes).strip()


def _extraer_montos_normalizados(texto: str) -> set[str]:
    return {
        re.sub(r"\s+", "", m.group(0)).rstrip(".,")
        for m in _PATRON_MONTO.finditer(texto or "")
    }


class GeneradorDialogoGemini:
    """
    Cliente resiliente para redaccion comercial con Gemini API v1beta.
    Nunca lanza excepciones hacia los agentes SPADE ni bloquea el flujo:
    ante cuota agotada (429), error de red o alucinacion de datos, devuelve
    el texto verificado base y activa enfriamiento automatico.
    """

    def __init__(
        self,
        api_key: str = "",
        modelo: str = "gemini-3-flash-preview",
        timeout_segundos: float = 4.0,
        http_client: Optional[httpx.Client] = None,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.modelo = (modelo or "gemini-3-flash-preview").strip()
        self.timeout_segundos = max(0.5, float(timeout_segundos or 4.0))
        self._http_client = http_client
        self._reloj = reloj
        self._bloqueado_hasta: float = 0.0
        self.ultimo_error: str = ""
        self.llamadas_exitosas: int = 0
        self.fallbacks_activados: int = 0

    @property
    def activo(self) -> bool:
        """Indica si existe una API key de Gemini configurada."""
        return bool(self.api_key)

    @property
    def en_enfriamiento(self) -> bool:
        """Indica si el circuit breaker esta abierto para evitar bloqueos."""
        return self._reloj() < self._bloqueado_hasta

    def estado_actual(self) -> Dict[str, Any]:
        """Retorna el estado operativo del generador para monitoreo y telemetria."""
        if not self.activo:
            modo = "Plantilla Local Verificada (sin GEMINI_API_KEY)"
        elif self.en_enfriamiento:
            modo = "Fallback Local (Enfriamiento activo)"
        else:
            modo = f"Gemini API ({self.modelo})"
        return {
            "habilitado": self.activo,
            "modelo": self.modelo,
            "modo": modo,
            "en_enfriamiento": self.en_enfriamiento,
            "llamadas_exitosas": self.llamadas_exitosas,
            "fallbacks_activados": self.fallbacks_activados,
            "ultimo_error": self.ultimo_error,
        }

    def _activar_enfriamiento(self, segundos: float, motivo: str) -> None:
        self._bloqueado_hasta = self._reloj() + segundos
        self.ultimo_error = motivo
        self.fallbacks_activados += 1

    @staticmethod
    def _validar_integridad(
        respuesta_llm: str,
        texto_base: str,
        hechos_obligatorios: Iterable[str],
    ) -> bool:
        limpia = (respuesta_llm or "").strip()
        if len(limpia) < 12:
            return False

        norm_resp = _normalizar_comparacion(limpia)
        if "complejidad 1-5" in norm_resp or "del 1 al 5" in norm_resp:
            return False

        for hecho in hechos_obligatorios:
            hecho_limpio = str(hecho or "").strip()
            if not hecho_limpio:
                continue
            if _normalizar_comparacion(hecho_limpio) not in norm_resp:
                return False

        montos_permitidos = _extraer_montos_normalizados(texto_base)
        for hecho in hechos_obligatorios:
            montos_permitidos.update(_extraer_montos_normalizados(str(hecho)))

        montos_llm = _extraer_montos_normalizados(limpia)
        if montos_llm - montos_permitidos:
            return False

        return True

    def redactar_respuesta(
        self,
        *,
        tipo_respuesta: str,
        texto_base: str,
        contexto_negocio: Optional[Dict[str, Any]] = None,
        mensaje_cliente: str = "",
        historial_reciente: Optional[List[Dict[str, Any]]] = None,
        hechos_obligatorios: Optional[Iterable[str]] = None,
    ) -> str:
        """
        Reescribe `texto_base` en tono humano y natural mediante Gemini.
        Si Gemini no esta configurado, esta en enfriamiento, falla o altera
        algun dato critico, retorna `texto_base` intacto.
        """
        if not self.activo or self.en_enfriamiento:
            return texto_base

        hechos = [str(h).strip() for h in (hechos_obligatorios or []) if str(h).strip()]
        negocio = contexto_negocio or {}
        nombre_negocio = str(negocio.get("nombre_negocio", "el negocio"))
        rubro = str(negocio.get("rubro", "servicios profesionales"))

        lineas_historial: List[str] = []
        for turno in (historial_reciente or [])[-4:]:
            rol = "Cliente" if turno.get("rol") == "cliente" else "Asesor"
            txt = str(turno.get("texto", "")).strip()
            if txt:
                lineas_historial.append(f"- {rol}: {txt[:220]}")

        bloque_hechos = (
            "\n".join(f"- {h}" for h in hechos)
            if hechos
            else "- Conservar fielmente toda la informacion del borrador."
        )
        prompt_usuario = (
            f"Negocio: {nombre_negocio} ({rubro})\n"
            f"Situacion comercial: {tipo_respuesta}\n"
            f"Ultimo mensaje del cliente: {mensaje_cliente or '(sin mensaje previo)'}\n"
            f"Historial reciente:\n"
            + ("\n".join(lineas_historial) if lineas_historial else "- (inicio de conversacion)")
            + f"\n\nBorrador verificado por el sistema multiagente:\n{texto_base}\n\n"
            f"Datos exactos que DEBEN aparecer literalmente:\n{bloque_hechos}\n\n"
            "Redacta la respuesta final para el cliente:"
        )

        url = f"{_URL_BASE_GEMINI}/{self.modelo}:generateContent"
        cuerpo = {
            "systemInstruction": {"parts": [{"text": _INSTRUCCION_SISTEMA}]},
            "contents": [{"role": "user", "parts": [{"text": prompt_usuario}]}],
            "generationConfig": {
                "temperature": 0.65,
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            if self._http_client is not None:
                resp = self._http_client.post(
                    url, headers=headers, json=cuerpo, timeout=self.timeout_segundos
                )
            else:
                with httpx.Client(timeout=self.timeout_segundos) as cliente:
                    resp = cliente.post(url, headers=headers, json=cuerpo)
        except (httpx.TimeoutException, httpx.HTTPError, OSError) as exc:
            self._activar_enfriamiento(30.0, f"error_red:{type(exc).__name__}")
            return texto_base

        if resp.status_code == 429:
            self._activar_enfriamiento(120.0, "cuota_o_tokens_agotados_429")
            return texto_base
        if resp.status_code in {401, 403}:
            self._activar_enfriamiento(600.0, f"api_key_invalida_{resp.status_code}")
            return texto_base
        if resp.status_code >= 400:
            self._activar_enfriamiento(30.0, f"http_{resp.status_code}")
            return texto_base

        try:
            data = resp.json()
            candidatos = data.get("candidates") or []
            partes = candidatos[0]["content"]["parts"]
            texto_generado = "".join(
                str(p.get("text", "")) for p in partes if isinstance(p, dict)
            ).strip()
        except (ValueError, KeyError, IndexError, TypeError):
            self._activar_enfriamiento(30.0, "respuesta_invalida")
            return texto_base

        if not self._validar_integridad(texto_generado, texto_base, hechos):
            self.ultimo_error = "validacion_hechos_fallback"
            self.fallbacks_activados += 1
            return texto_base

        self.llamadas_exitosas += 1
        self.ultimo_error = ""
        return texto_generado
