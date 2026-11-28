"""
Servidor FastAPI multi-negocio:
- Webhooks para Botpress Studio y Meta WhatsApp Cloud API directa
- Panel PWA movil de alto contraste para Super-Admin y Duenos de Negocio
- Onboarding conversacional con Google Gemini Flash Structured Outputs
"""
from __future__ import annotations

import hmac
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.comercial.configuracion import ConfiguracionComercial
from src.comercial.herramientas_negocio import construir_cliente_calendar_negocio
from src.comercial.motor_gemini import AsistenteOnboardingGemini, OrquestadorComercialGemini
from src.comercial.notificador_whatsapp import (
    enviar_mensaje_whatsapp_cloud,
    es_phone_number_id_valido,
    verificar_firma_meta,
)
from src.comercial.repositorio_multitenant import (
    ConflictoAgendaError,
    ConflictoCanalError,
    RepositorioComercialSQLite,
    RepositorioComercialSupabase,
    RepositorioMultiTenant,
    generar_token_sesion,
    normalizar_telefono,
    pin_fingerprint,
    resolver_zona_horaria_segura,
    sanitizar_negocio_publico,
    validar_horario_laboral,
    validar_token_sesion,
    validar_zona_horaria,
    verificar_pin,
)
from src.persistencia.google_calendar import ClienteGoogleCalendar, ErrorGoogleCalendar

_DIR_STATIC = Path(__file__).resolve().parent / "static"

_CAMPOS_PERMITIDOS_DUENO = {
    "nombre_negocio",
    "rubro",
    "moneda",
    "unidad_cantidad",
    "telefono_dueno",
    "google_calendar_id",
    "zona_horaria",
    "mensaje_catalogo_intro",
    "horario_laboral",
    "reglas_y_faq",
    "pin_dueno",
}

_CAMPOS_SOLO_SUPERADMIN = {
    "whatsapp_negocio",
    "whatsapp_phone_number_id",
    "whatsapp_access_token",
    "botpress_bot_id",
    "margen_base",
    "margen_minimo",
    "descuento_maximo",
    "activo",
}


class _RateLimiterLogin:
    """Limitador simple en memoria contra fuerza bruta en /api/auth/login."""

    def __init__(self, max_intentos: int = 6, ventana_seg: int = 300) -> None:
        self._max = max_intentos
        self._ventana = ventana_seg
        self._lock = threading.Lock()
        self._fallos: Dict[str, List[float]] = defaultdict(list)

    def verificar_permitido(self, clave: str) -> None:
        ahora = time.monotonic()
        with self._lock:
            recientes = [t for t in self._fallos.get(clave, []) if ahora - t < self._ventana]
            self._fallos[clave] = recientes
            if len(recientes) >= self._max:
                raise HTTPException(
                    status_code=429,
                    detail="Demasiados intentos fallidos. Espera unos minutos antes de reintentar.",
                )

    def registrar_fallo(self, clave: str) -> None:
        ahora = time.monotonic()
        with self._lock:
            self._fallos[clave].append(ahora)

    def limpiar(self, clave: str) -> None:
        with self._lock:
            self._fallos.pop(clave, None)


# ---------------------------------------------------------------------------
# Modelos de Peticion HTTP
# ---------------------------------------------------------------------------
class PeticionLogin(BaseModel):
    rol: str = Field(description="'superadmin' o 'dueno'")
    clave: str = Field(default="", description="Clave maestra superadmin o PIN del dueno")
    negocio_id: Optional[str] = None


class PeticionWebhookBotpress(BaseModel):
    negocio_id: Optional[str] = None
    botpress_bot_id: Optional[str] = None
    whatsapp_negocio: Optional[str] = None
    phone_number_id: Optional[str] = None
    canal: Optional[str] = None
    channel: Optional[str] = None
    id_cliente: Optional[str] = None
    userId: Optional[str] = None
    conversationId: Optional[str] = None
    nombre: Optional[str] = None
    userName: Optional[str] = None
    texto: Optional[str] = None
    text: Optional[str] = None
    mensaje: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PeticionServicio(BaseModel):
    clave: Optional[str] = None
    nombre: str
    descripcion: str = ""
    tarifa_unitaria: float = Field(ge=0)
    duracion_minutos: int = Field(default=60, ge=15, le=720)
    palabras_clave: List[str] = Field(default_factory=list)
    activo: bool = True


class PeticionPromocion(BaseModel):
    codigo: str
    descripcion: str
    descuento_max: float = Field(ge=0, le=90)
    activa: bool = True
    condiciones: Dict[str, Any] = Field(default_factory=dict)


class PeticionNegocio(BaseModel):
    id: Optional[str] = None
    nombre_negocio: str
    rubro: str
    moneda: str = "COP"
    unidad_cantidad: str = "servicios"
    telefono_dueno: str = ""
    whatsapp_negocio: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_access_token: Optional[str] = None
    botpress_bot_id: str = ""
    google_calendar_id: str = ""
    zona_horaria: str = "America/Bogota"
    margen_base: float = Field(default=0.35, ge=0.0, le=1.0)
    margen_minimo: float = Field(default=0.22, ge=0.0, le=1.0)
    descuento_maximo: float = Field(default=15.0, ge=0.0, le=90.0)
    mensaje_catalogo_intro: str = "Con gusto te comparto nuestros servicios"
    activo: bool = True
    horario_laboral: Dict[str, Any] = Field(default_factory=dict)
    reglas_y_faq: Dict[str, Any] = Field(default_factory=dict)
    pin_dueno: Optional[str] = None
    servicios: Optional[List[PeticionServicio]] = None
    promociones: Optional[List[PeticionPromocion]] = None

    @field_validator("zona_horaria")
    @classmethod
    def _val_tz(cls, v: str) -> str:
        return validar_zona_horaria(v)

    @field_validator("horario_laboral")
    @classmethod
    def _val_horario(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            return {}
        return validar_horario_laboral(v)


class PeticionActualizarNegocio(BaseModel):
    """Esquema parcial: evita sobreescribir campos no enviados desde el panel PWA."""

    nombre_negocio: Optional[str] = None
    rubro: Optional[str] = None
    moneda: Optional[str] = None
    unidad_cantidad: Optional[str] = None
    telefono_dueno: Optional[str] = None
    whatsapp_negocio: Optional[str] = None
    whatsapp_phone_number_id: Optional[str] = None
    whatsapp_access_token: Optional[str] = None
    botpress_bot_id: Optional[str] = None
    google_calendar_id: Optional[str] = None
    zona_horaria: Optional[str] = None
    margen_base: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    margen_minimo: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    descuento_maximo: Optional[float] = Field(default=None, ge=0.0, le=90.0)
    mensaje_catalogo_intro: Optional[str] = None
    activo: Optional[bool] = None
    horario_laboral: Optional[Dict[str, Any]] = None
    reglas_y_faq: Optional[Dict[str, Any]] = None
    pin_dueno: Optional[str] = None

    @field_validator("zona_horaria")
    @classmethod
    def _val_tz(cls, v: Optional[str]) -> Optional[str]:
        return validar_zona_horaria(v) if v is not None else None

    @field_validator("horario_laboral")
    @classmethod
    def _val_horario(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        return validar_horario_laboral(v)


class PeticionCitaManual(BaseModel):
    fecha_hora: str = Field(description="YYYY-MM-DD HH:MM")
    duracion_minutos: int = Field(default=60, ge=15, le=720)
    servicio: str = "Bloqueo / Cita Manual"
    nombre_cliente: str = "Reserva Manual"
    telefono_cliente: str = ""
    observaciones: str = ""


class PeticionProbarChat(BaseModel):
    id_cliente: str = "simulador_panel"
    nombre: str = "Cliente Simulador"
    texto: str


class PeticionOnboarding(BaseModel):
    historial: List[Dict[str, str]] = Field(default_factory=list)
    mensaje: str
    borrador_actual: Optional[Dict[str, Any]] = None


def crear_repositorio_comercial(config: ConfiguracionComercial) -> RepositorioMultiTenant:
    if config.usa_supabase:
        return RepositorioComercialSupabase(config.supabase_url, config.supabase_key)
    return RepositorioComercialSQLite(config.sqlite_db_path)


def _extraer_mensajes_meta_whatsapp(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extrae mensajes de texto entrantes desde el formato oficial de Meta Cloud API."""
    mensajes: List[Dict[str, str]] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            metadata = value.get("metadata") or {}
            phone_number_id = str(metadata.get("phone_number_id") or "")
            display_phone = str(metadata.get("display_phone_number") or "")
            contactos = {
                str(c.get("wa_id") or ""): str((c.get("profile") or {}).get("name") or "Cliente WhatsApp")
                for c in (value.get("contacts") or [])
            }
            for msg in value.get("messages") or []:
                if msg.get("type") != "text":
                    continue
                remitente = str(msg.get("from") or "")
                texto = str((msg.get("text") or {}).get("body") or "").strip()
                if remitente and texto:
                    mensajes.append(
                        {
                            "phone_number_id": phone_number_id,
                            "display_phone": display_phone,
                            "from": remitente,
                            "nombre": contactos.get(remitente, "Cliente WhatsApp"),
                            "texto": texto,
                        }
                    )
    return mensajes


def crear_app_comercial(
    config: Optional[ConfiguracionComercial] = None,
    repositorio: Optional[RepositorioMultiTenant] = None,
    orquestador: Optional[OrquestadorComercialGemini] = None,
    onboarding: Optional[AsistenteOnboardingGemini] = None,
    enviador_whatsapp: Callable[..., Dict[str, Any]] = enviar_mensaje_whatsapp_cloud,
    fabrica_calendar: Optional[
        Callable[[Dict[str, Any], ConfiguracionComercial], Optional[ClienteGoogleCalendar]]
    ] = None,
) -> FastAPI:
    cfg = config or ConfiguracionComercial.desde_entorno()
    repo = repositorio or crear_repositorio_comercial(cfg)
    fab_cal = fabrica_calendar or construir_cliente_calendar_negocio
    orq = orquestador or OrquestadorComercialGemini(repo, cfg, fabrica_calendar=fab_cal)
    onb = onboarding or AsistenteOnboardingGemini(cfg)
    rate_limiter = _RateLimiterLogin()

    app = FastAPI(
        title="Plataforma Comercial Multi-Negocio",
        version="3.0.0",
    )

    @app.middleware("http")
    async def agregar_cabeceras_seguridad(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "script-src 'self'; "
            "connect-src 'self'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'",
        )
        return response

    if _DIR_STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_DIR_STATIC)), name="static")

    # -----------------------------------------------------------------------
    # Helpers de Seguridad y Aislamiento Multi-Tenant
    # -----------------------------------------------------------------------
    def _validar_api_key_botpress(
        x_botpress_key: Optional[str],
        x_api_key: Optional[str],
        authorization: Optional[str],
    ) -> None:
        if not cfg.botpress_api_key:
            if cfg.allow_insecure_webhooks:
                return
            raise HTTPException(
                status_code=503,
                detail="Webhook deshabilitado: configura BOTPRESS_API_KEY en .env.",
            )
        token = (x_botpress_key or x_api_key or "").strip()
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token or not hmac.compare_digest(cfg.botpress_api_key, token):
            raise HTTPException(status_code=401, detail="API Key de webhook invalida.")

    def _requerir_sesion(
        authorization: Optional[str] = Header(default=None),
        x_admin_api_key: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        if x_admin_api_key and hmac.compare_digest(cfg.admin_access_key, x_admin_api_key):
            return {"rol": "superadmin", "negocio_id": None}
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            sesion = validar_token_sesion(cfg.session_secret, token)
            if sesion:
                if sesion.get("rol") == "dueno":
                    neg = repo.obtener_negocio(str(sesion.get("negocio_id") or ""))
                    if not neg or not neg.get("activo", True):
                        raise HTTPException(status_code=401, detail="Negocio inactivo o sesion revocada.")
                    fp_actual = pin_fingerprint(str(neg.get("pin_dueno_hash") or ""))
                    fp_token = str(sesion.get("pin_fp") or "")
                    if not fp_actual or not hmac.compare_digest(fp_actual, fp_token):
                        raise HTTPException(status_code=401, detail="Sesion revocada por cambio de PIN.")
                return sesion
        raise HTTPException(status_code=401, detail="Sesion invalida o expirada.")

    def _requerir_superadmin(sesion: Dict[str, Any] = Depends(_requerir_sesion)) -> Dict[str, Any]:
        if sesion.get("rol") != "superadmin":
            raise HTTPException(status_code=403, detail="Requiere permisos de Super-Admin.")
        return sesion

    def _verificar_acceso_negocio(sesion: Dict[str, Any], negocio_id: str) -> None:
        if sesion.get("rol") == "superadmin":
            return
        if sesion.get("rol") == "dueno" and sesion.get("negocio_id") == negocio_id:
            return
        raise HTTPException(
            status_code=403, detail="No tienes permisos para acceder a este negocio."
        )

    def _notificar_dueno_si_aplica(negocio: Dict[str, Any], resultado: Dict[str, Any]) -> None:
        if not resultado.get("notificar_dueno"):
            return
        token_wa = str(negocio.get("whatsapp_access_token") or cfg.whatsapp_default_access_token or "")
        phone_id = str(negocio.get("whatsapp_phone_number_id") or "")
        tel_dueno = str(resultado.get("telefono_dueno") or negocio.get("telefono_dueno") or "")
        msg_dueno = str(resultado.get("mensaje_notificacion_dueno") or "")
        if token_wa and es_phone_number_id_valido(phone_id) and tel_dueno and msg_dueno:
            enviador_whatsapp(
                phone_number_id=phone_id,
                access_token=token_wa,
                destinatario=tel_dueno,
                texto=msg_dueno,
            )

    def _resolver_y_procesar_botpress(
        req: PeticionWebhookBotpress, negocio_ruta: Optional[str] = None
    ) -> Dict[str, Any]:
        neg_id = (negocio_ruta or req.negocio_id or "").strip()
        if not neg_id:
            meta = req.metadata or {}
            encontrado = repo.buscar_negocio_por_canal(
                phone_number_id=req.phone_number_id or str(meta.get("phone_number_id") or ""),
                whatsapp_negocio=req.whatsapp_negocio or str(meta.get("whatsapp_negocio") or ""),
                botpress_bot_id=req.botpress_bot_id or str(meta.get("bot_id") or ""),
            )
            neg_id = encontrado["id"] if encontrado else cfg.negocio_id_default

        negocio = repo.obtener_negocio(neg_id)
        if not negocio:
            raise HTTPException(status_code=404, detail=f"Negocio '{neg_id}' no encontrado.")

        canal = (req.canal or req.channel or "WhatsApp").strip()
        id_cliente = (req.id_cliente or req.userId or req.conversationId or "").strip()
        texto = (req.texto or req.text or req.mensaje or req.message or "").strip()
        nombre = (req.nombre or req.userName or f"{canal} Cliente").strip()
        if not id_cliente or not texto:
            raise HTTPException(status_code=400, detail="Se requieren 'id_cliente' y 'texto'.")

        resultado = orq.procesar_mensaje(
            negocio_id=neg_id,
            id_cliente=id_cliente,
            texto=texto,
            canal=canal,
            nombre_cliente=nombre,
        )
        _notificar_dueno_si_aplica(negocio, resultado)
        return resultado

    def _procesar_lote_meta_whatsapp(payload: Dict[str, Any]) -> Dict[str, Any]:
        entrantes = _extraer_mensajes_meta_whatsapp(payload)
        procesados = []
        for item in entrantes:
            negocio = repo.buscar_negocio_por_canal(
                phone_number_id=item["phone_number_id"],
                whatsapp_negocio=item["display_phone"],
            )
            if not negocio:
                continue
            res = orq.procesar_mensaje(
                negocio_id=negocio["id"],
                id_cliente=item["from"],
                texto=item["texto"],
                canal="WhatsApp",
                nombre_cliente=item["nombre"],
            )
            token_wa = str(negocio.get("whatsapp_access_token") or cfg.whatsapp_default_access_token or "")
            # Priorizar el phone_number_id confiable registrado en BD del negocio
            phone_id_confiable = str(negocio.get("whatsapp_phone_number_id") or item["phone_number_id"] or "").strip()
            if (
                res.get("responder_al_cliente")
                and res.get("respuesta")
                and token_wa
                and es_phone_number_id_valido(phone_id_confiable)
            ):
                enviador_whatsapp(
                    phone_number_id=phone_id_confiable,
                    access_token=token_wa,
                    destinatario=item["from"],
                    texto=res["respuesta"],
                )
            _notificar_dueno_si_aplica(negocio, res)
            procesados.append(
                {
                    "negocio_id": negocio["id"],
                    "id_cliente": item["from"],
                    "pausado_humano": res.get("pausado_humano", False),
                }
            )
        return {"status": "ok", "procesados": len(procesados), "detalle": procesados}

    # -----------------------------------------------------------------------
    # PWA Frontend
    # -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def pwa_index() -> str:
        ruta = _DIR_STATIC / "index.html"
        if not ruta.exists():
            return "<h1>Panel Comercial</h1>"
        return ruta.read_text(encoding="utf-8")

    @app.get("/manifest.json")
    def pwa_manifest():
        return FileResponse(str(_DIR_STATIC / "manifest.json"), media_type="application/manifest+json")

    @app.get("/sw.js")
    def pwa_sw():
        return FileResponse(
            str(_DIR_STATIC / "sw.js"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/health")
    def healthcheck() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": "3.0.0-comercial",
            "modo_bd": repo.modo_bd,
            "motor_ia": cfg.gemini_model,
            "gemini_activo": orq.activo,
            "negocios_registrados": len(repo.listar_negocios()),
        }

    # -----------------------------------------------------------------------
    # Autenticacion (Super-Admin y Dueno por Negocio)
    # -----------------------------------------------------------------------
    @app.get("/api/public/negocios-acceso")
    def listar_negocios_publicos() -> Dict[str, Any]:
        """Lista minima publica (id, nombre, rubro) para el selector de ingreso de duenos."""
        items = [
            {"id": n["id"], "nombre_negocio": n["nombre_negocio"], "rubro": n["rubro"]}
            for n in repo.listar_negocios()
            if n.get("activo", True)
        ]
        return {"negocios": items}

    @app.post("/api/auth/login")
    def iniciar_sesion(req: PeticionLogin, request: Request) -> Dict[str, Any]:
        rol = req.rol.strip().lower()
        ip_cliente = request.client.host if request.client else "local"
        clave_rl = f"{ip_cliente}:{rol}:{(req.negocio_id or '').strip()}"
        rate_limiter.verificar_permitido(clave_rl)

        if rol == "superadmin":
            if not hmac.compare_digest(cfg.admin_access_key, req.clave.strip()):
                rate_limiter.registrar_fallo(clave_rl)
                raise HTTPException(status_code=401, detail="Clave de Super-Admin incorrecta.")
            rate_limiter.limpiar(clave_rl)
            token = generar_token_sesion(cfg.session_secret, rol="superadmin")
            return {"token": token, "rol": "superadmin", "negocio_id": None}

        if rol == "dueno":
            neg_id = (req.negocio_id or "").strip()
            neg = repo.obtener_negocio(neg_id)
            if not neg:
                rate_limiter.registrar_fallo(clave_rl)
                raise HTTPException(status_code=404, detail="Negocio no encontrado.")
            if not neg.get("activo", True):
                raise HTTPException(status_code=403, detail="Este negocio se encuentra desactivado.")
            if not neg.get("pin_dueno_hash"):
                raise HTTPException(
                    status_code=403,
                    detail="Este negocio aun no tiene un PIN de dueno configurado.",
                )
            if not verificar_pin(req.clave, neg["pin_dueno_hash"]):
                rate_limiter.registrar_fallo(clave_rl)
                raise HTTPException(status_code=401, detail="PIN del negocio incorrecto.")
            rate_limiter.limpiar(clave_rl)
            token = generar_token_sesion(
                cfg.session_secret,
                rol="dueno",
                negocio_id=neg_id,
                pin_fp=pin_fingerprint(neg["pin_dueno_hash"]),
            )
            return {
                "token": token,
                "rol": "dueno",
                "negocio_id": neg_id,
                "nombre_negocio": neg["nombre_negocio"],
            }

        raise HTTPException(status_code=400, detail="Rol no valido. Usa 'superadmin' o 'dueno'.")

    # -----------------------------------------------------------------------
    # Webhooks Multi-Canal (Botpress + Meta WhatsApp Cloud API Directa)
    # -----------------------------------------------------------------------
    @app.post("/api/webhook/{negocio_id}")
    def webhook_por_negocio(
        negocio_id: str,
        req: PeticionWebhookBotpress,
        x_botpress_key: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _validar_api_key_botpress(x_botpress_key, x_api_key, authorization)
        return _resolver_y_procesar_botpress(req, negocio_ruta=negocio_id)

    @app.post("/api/botpress/mensaje")
    def webhook_botpress_compatible(
        req: PeticionWebhookBotpress,
        x_botpress_key: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _validar_api_key_botpress(x_botpress_key, x_api_key, authorization)
        return _resolver_y_procesar_botpress(req)

    @app.get("/api/whatsapp/webhook")
    def meta_whatsapp_verificacion(
        hub_mode: str = Query(default="", alias="hub.mode"),
        hub_verify_token: str = Query(default="", alias="hub.verify_token"),
        hub_challenge: str = Query(default="", alias="hub.challenge"),
    ):
        if hub_mode == "subscribe" and hmac.compare_digest(
            cfg.whatsapp_verify_token, hub_verify_token
        ):
            return PlainTextResponse(content=hub_challenge, status_code=200)
        raise HTTPException(status_code=403, detail="Token de verificacion WhatsApp invalido.")

    @app.post("/api/whatsapp/webhook")
    async def meta_whatsapp_entrante(
        request: Request,
        x_hub_signature_256: str = Header(default=""),
    ) -> Dict[str, Any]:
        cuerpo_raw = await request.body()
        if not cfg.whatsapp_app_secret and not cfg.allow_insecure_webhooks:
            raise HTTPException(
                status_code=503,
                detail="Webhook Meta deshabilitado: configura WHATSAPP_APP_SECRET en .env.",
            )
        if not verificar_firma_meta(
            cuerpo_raw,
            x_hub_signature_256,
            cfg.whatsapp_app_secret,
            permitir_sin_secreto=cfg.allow_insecure_webhooks,
        ):
            raise HTTPException(status_code=401, detail="Firma X-Hub-Signature-256 invalida.")
        try:
            payload = json.loads(cuerpo_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPException(status_code=400, detail="Cuerpo JSON de webhook invalido.")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="El payload del webhook debe ser un objeto JSON.")
        return await run_in_threadpool(_procesar_lote_meta_whatsapp, payload)

    # -----------------------------------------------------------------------
    # API Super-Admin (Gestion de Todos los Negocios)
    # -----------------------------------------------------------------------
    @app.get("/api/admin/negocios")
    def admin_listar_negocios(
        _: Dict[str, Any] = Depends(_requerir_superadmin),
    ) -> Dict[str, Any]:
        negocios = repo.listar_negocios()
        enriquecidos = []
        for neg in negocios:
            nid = neg["id"]
            pausados = repo.listar_perfiles(nid, solo_pausados=True)
            alertas = repo.listar_alertas(nid, solo_abiertas=True)
            servicios = repo.listar_servicios(nid, solo_activos=True)
            citas = repo.listar_agendamientos(nid, solo_activos=True)
            enriquecidos.append(
                {
                    **sanitizar_negocio_publico(neg),
                    "metricas": {
                        "chats_pausados": len(pausados),
                        "alertas_abiertas": len(alertas),
                        "servicios_activos": len(servicios),
                        "citas_activas": len(citas),
                    },
                }
            )
        return {"negocios": enriquecidos, "motor_ia": cfg.gemini_model, "gemini_activo": orq.activo}

    @app.post("/api/admin/negocios")
    def admin_crear_o_actualizar_negocio(
        req: PeticionNegocio,
        _: Dict[str, Any] = Depends(_requerir_superadmin),
    ) -> Dict[str, Any]:
        datos = req.model_dump(exclude={"servicios", "promociones"}, exclude_none=True)
        try:
            guardado = repo.guardar_negocio(datos)
        except ConflictoCanalError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        nid = guardado["id"]
        if req.servicios is not None:
            for idx, srv in enumerate(req.servicios, start=1):
                repo.guardar_servicio(nid, {**srv.model_dump(), "servicio_id_num": idx})
        if req.promociones is not None:
            for promo in req.promociones:
                repo.guardar_promocion(nid, promo.model_dump())
        return {
            "status": "ok",
            "negocio": sanitizar_negocio_publico(guardado),
            "servicios": repo.listar_servicios(nid),
            "promociones": repo.listar_promociones(nid),
        }

    @app.delete("/api/admin/negocios/{negocio_id}")
    def admin_eliminar_negocio(
        negocio_id: str,
        _: Dict[str, Any] = Depends(_requerir_superadmin),
    ) -> Dict[str, Any]:
        ok = repo.eliminar_negocio(negocio_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Negocio no encontrado.")
        return {"status": "ok", "eliminado": negocio_id}

    # -----------------------------------------------------------------------
    # API Operativa por Negocio (Aislada para el Dueno y accesible a Super-Admin)
    # -----------------------------------------------------------------------
    @app.get("/api/negocios/{negocio_id}/resumen")
    def resumen_negocio(
        negocio_id: str,
        request: Request,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        neg = repo.obtener_negocio(negocio_id)
        if not neg:
            raise HTTPException(status_code=404, detail="Negocio no encontrado.")

        pausados = repo.listar_perfiles(negocio_id, solo_pausados=True)
        alertas = repo.listar_alertas(negocio_id, solo_abiertas=False)
        servicios = repo.listar_servicios(negocio_id)
        promos = repo.listar_promociones(negocio_id)
        citas = repo.listar_agendamientos(negocio_id)

        alertas_por_cli: Dict[str, Dict[str, Any]] = {}
        for a in alertas:
            if not a.get("resuelta") and a["id_cliente"] not in alertas_por_cli:
                alertas_por_cli[a["id_cliente"]] = a

        chats_pausados_vista = []
        for p in pausados:
            al = alertas_por_cli.get(p["id_cliente"], {})
            tel = normalizar_telefono(p["id_cliente"])
            chats_pausados_vista.append(
                {
                    "id_cliente": p["id_cliente"],
                    "nombre": p.get("nombre", "Cliente"),
                    "canal": p.get("canal", "WhatsApp"),
                    "motivo": al.get("motivo_critico", "Atencion humana solicitada"),
                    "resumen_ia": al.get("resumen_ia", ""),
                    "timestamp": al.get("timestamp", p.get("updated_at", "")),
                    "historial": p.get("historial_interacciones", [])[-10:],
                    "ultima_cotizacion": p.get("ultima_cotizacion"),
                    "enlace_whatsapp": f"https://wa.me/{tel}" if len(tel) >= 10 else "",
                }
            )

        base_url = cfg.public_base_url or str(request.base_url).rstrip("/")
        return {
            "negocio": sanitizar_negocio_publico(neg),
            "kpis": {
                "chats_pausados": len(chats_pausados_vista),
                "citas_activas": sum(1 for c in citas if c.get("estado") in ("confirmada", "pendiente")),
                "servicios_activos": sum(1 for s in servicios if s.get("activo", True)),
                "promociones_activas": sum(1 for pr in promos if pr.get("activa", True)),
            },
            "chats_pausados": chats_pausados_vista,
            "alertas": alertas[:30],
            "agendamientos": citas,
            "servicios": servicios,
            "promociones": promos,
            "integracion": {
                "webhook_botpress": f"{base_url}/api/webhook/{negocio_id}",
                "webhook_whatsapp_meta": f"{base_url}/api/whatsapp/webhook",
                "gemini_activo": orq.activo,
                "motor_ia": cfg.gemini_model,
            },
        }

    @app.put("/api/negocios/{negocio_id}/configuracion")
    def actualizar_configuracion_negocio(
        negocio_id: str,
        req: PeticionActualizarNegocio,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        existente = repo.obtener_negocio(negocio_id)
        if not existente:
            raise HTTPException(status_code=404, detail="Negocio no encontrado.")

        datos = req.model_dump(exclude_unset=True, exclude_none=True)
        if sesion.get("rol") != "superadmin":
            intentados_restringidos = set(datos.keys()) & _CAMPOS_SOLO_SUPERADMIN
            if intentados_restringidos:
                raise HTTPException(
                    status_code=403,
                    detail="Solo Super-Admin puede modificar canales WhatsApp/Botpress, estado activo o margenes.",
                )
            datos = {k: v for k, v in datos.items() if k in _CAMPOS_PERMITIDOS_DUENO}

        datos["id"] = negocio_id
        try:
            guardado = repo.guardar_negocio(datos)
        except ConflictoCanalError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        resp_payload: Dict[str, Any] = {"status": "ok", "negocio": sanitizar_negocio_publico(guardado)}
        # Si el propio dueno roto su PIN, emitir nuevo token para que no pierda su sesion actual
        if sesion.get("rol") == "dueno" and datos.get("pin_dueno"):
            resp_payload["nuevo_token"] = generar_token_sesion(
                cfg.session_secret,
                rol="dueno",
                negocio_id=negocio_id,
                pin_fp=pin_fingerprint(guardado.get("pin_dueno_hash", "")),
            )
        return resp_payload

    @app.post("/api/negocios/{negocio_id}/servicios")
    def guardar_servicio_negocio(
        negocio_id: str,
        req: PeticionServicio,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        srv = repo.guardar_servicio(negocio_id, req.model_dump())
        return {"status": "ok", "servicio": srv}

    @app.delete("/api/negocios/{negocio_id}/servicios/{clave}")
    def eliminar_servicio_negocio(
        negocio_id: str,
        clave: str,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        if not repo.eliminar_servicio(negocio_id, clave):
            raise HTTPException(status_code=404, detail="Servicio no encontrado.")
        return {"status": "ok", "eliminado": clave}

    @app.post("/api/negocios/{negocio_id}/promociones")
    def guardar_promocion_negocio(
        negocio_id: str,
        req: PeticionPromocion,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        promo = repo.guardar_promocion(negocio_id, req.model_dump())
        return {"status": "ok", "promocion": promo}

    @app.delete("/api/negocios/{negocio_id}/promociones/{codigo}")
    def eliminar_promocion_negocio(
        negocio_id: str,
        codigo: str,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        if not repo.eliminar_promocion(negocio_id, codigo):
            raise HTTPException(status_code=404, detail="Promocion no encontrada.")
        return {"status": "ok", "eliminado": codigo}

    @app.post("/api/negocios/{negocio_id}/chats/{id_cliente}/reactivar")
    def reactivar_chat_negocio(
        negocio_id: str,
        id_cliente: str,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        perfil = repo.obtener_perfil(negocio_id, id_cliente)
        if not perfil:
            raise HTTPException(status_code=404, detail="Chat de cliente no encontrado en este negocio.")
        perfil["automatizacion_activa"] = True
        perfil["tono_actual"] = "neutral"
        repo.guardar_perfil(negocio_id, perfil)
        repo.resolver_alertas_cliente(negocio_id, id_cliente)
        return {"status": "ok", "negocio_id": negocio_id, "id_cliente": id_cliente, "automatizacion_activa": True}

    @app.post("/api/negocios/{negocio_id}/agendamientos")
    def crear_cita_manual(
        negocio_id: str,
        req: PeticionCitaManual,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        neg = repo.obtener_negocio(negocio_id)
        if not neg:
            raise HTTPException(status_code=404, detail="Negocio no encontrado.")
        tz = resolver_zona_horaria_segura(str(neg.get("zona_horaria") or "America/Bogota"))
        try:
            ini = datetime.fromisoformat(req.fecha_hora.strip().replace("Z", "+00:00"))
            ini = ini.replace(tzinfo=tz) if ini.tzinfo is None else ini.astimezone(tz)
        except ValueError:
            raise HTTPException(status_code=400, detail="Usa formato YYYY-MM-DD HH:MM.")
        fin = ini + timedelta(minutes=req.duracion_minutos)
        try:
            cita = repo.registrar_agendamiento(
                negocio_id,
                {
                    "id_cliente": normalizar_telefono(req.telefono_cliente) or "panel_dueno",
                    "nombre_cliente": req.nombre_cliente,
                    "canal": "Panel",
                    "servicio": req.servicio,
                    "fecha_inicio": ini.isoformat(),
                    "fecha_fin": fin.isoformat(),
                    "estado": "confirmada",
                    "observaciones": req.observaciones,
                },
            )
        except ConflictoAgendaError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        cal_cliente = fab_cal(neg, cfg)
        if cal_cliente is not None:
            cal_cliente.duracion_minutos = req.duracion_minutos
            try:
                creado = cal_cliente.crear_evento(
                    fecha_hora=ini.strftime("%Y-%m-%d %H:%M"),
                    cliente=req.nombre_cliente,
                    titulo=f"{req.servicio} - {req.nombre_cliente}",
                    descripcion=req.observaciones or f"Reserva manual registrada desde panel ({neg['nombre_negocio']}).",
                )
                actualizada = repo.actualizar_agendamiento(
                    negocio_id,
                    str(cita["id"]),
                    {
                        "registrado_en": "supabase_y_google_calendar",
                        "google_calendar_estado": "sincronizado",
                        "google_event_id": creado.get("evento_google_id") or creado.get("google_event_id"),
                        "google_event_link": creado.get("html_link") or creado.get("google_event_link"),
                    },
                )
                if actualizada:
                    cita = actualizada
            except ErrorGoogleCalendar:
                actualizada = repo.actualizar_agendamiento(
                    negocio_id,
                    str(cita["id"]),
                    {"google_calendar_estado": "error"},
                )
                if actualizada:
                    cita = actualizada

        return {"status": "ok", "agendamiento": cita}

    @app.delete("/api/negocios/{negocio_id}/agendamientos/{id_agendamiento}")
    def cancelar_cita_negocio(
        negocio_id: str,
        id_agendamiento: str,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        cita = repo.obtener_agendamiento(negocio_id, id_agendamiento)
        if not cita or not repo.cancelar_agendamiento(negocio_id, id_agendamiento):
            raise HTTPException(status_code=404, detail="Agendamiento no encontrado.")

        evento_id = str(cita.get("google_event_id") or "").strip()
        if evento_id:
            neg = repo.obtener_negocio(negocio_id)
            cal_cliente = fab_cal(neg, cfg) if neg else None
            if cal_cliente is not None:
                try:
                    cal_cliente.eliminar_evento(evento_id)
                except ErrorGoogleCalendar:
                    pass
        return {"status": "ok", "cancelado": id_agendamiento}

    @app.post("/api/negocios/{negocio_id}/probar-chat")
    def probar_chat_negocio(
        negocio_id: str,
        req: PeticionProbarChat,
        sesion: Dict[str, Any] = Depends(_requerir_sesion),
    ) -> Dict[str, Any]:
        _verificar_acceso_negocio(sesion, negocio_id)
        try:
            return orq.procesar_mensaje(
                negocio_id=negocio_id,
                id_cliente=req.id_cliente,
                texto=req.texto,
                canal="SimuladorPWA",
                nombre_cliente=req.nombre,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Negocio no encontrado.")

    # -----------------------------------------------------------------------
    # Onboarding Conversacional con Google Gemini Flash (Solo Super-Admin)
    # -----------------------------------------------------------------------
    @app.post("/api/onboarding/chat")
    def onboarding_conversacional(
        req: PeticionOnboarding,
        _: Dict[str, Any] = Depends(_requerir_superadmin),
    ) -> Dict[str, Any]:
        return onb.procesar_turno(
            historial=req.historial,
            mensaje_usuario=req.mensaje,
            borrador_actual=req.borrador_actual,
        )

    return app
