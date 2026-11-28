"""
Motor conversacional y de onboarding basado en Google Gemini 2.5 Flash (`google-genai`).
Usa Tool Calling manual controlado para garantizar cero alucinaciones en precios/agenda
y Structured Outputs (Pydantic) para crear o actualizar negocios conversacionalmente.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.comercial.configuracion import ConfiguracionComercial
from src.comercial.herramientas_negocio import ContextoHerramientasNegocio
from src.comercial.repositorio_multitenant import (
    RepositorioMultiTenant,
    resolver_zona_horaria_segura,
)
from src.persistencia.google_calendar import ClienteGoogleCalendar

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


def _crear_cliente_genai(api_key: str) -> Any:
    """Construye el cliente Gemini con timeout HTTP seguro de 20 segundos."""
    if genai is None or not api_key:
        return None
    if genai_types is not None and hasattr(genai_types, "HttpOptions"):
        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=20_000),
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Esquemas Pydantic para Onboarding Conversacional Estructurado
# ---------------------------------------------------------------------------
class ServicioBorrador(BaseModel):
    clave: str = Field(default="", description="Identificador snake_case del servicio.")
    nombre: str = Field(description="Nombre comercial claro del servicio o producto.")
    descripcion: str = Field(default="", description="Que incluye exactamente el servicio.")
    tarifa_unitaria: float = Field(default=0.0, description="Precio numerico sin simbolos.")
    duracion_minutos: int = Field(default=60, description="Duracion aproximada en minutos.")


class PromocionBorrador(BaseModel):
    codigo: str = Field(description="Codigo corto en mayusculas, ej. COMBO_10.")
    descripcion: str = Field(description="Regla o beneficio de la promocion.")
    descuento_max: float = Field(default=10.0, description="Porcentaje maximo de descuento.")
    cantidad_minima: int = Field(default=2, description="Cantidad minima para aplicar.")


class PreguntaFrecuenteBorrador(BaseModel):
    pregunta: str
    respuesta: str


class HorarioBorrador(BaseModel):
    dias: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 6], description="1=Lunes .. 7=Domingo")
    inicio: str = Field(default="08:00", description="Hora apertura HH:MM")
    fin: str = Field(default="18:00", description="Hora cierre HH:MM")
    duracion_min: int = Field(default=60, description="Duracion base de citas en minutos")


class ReglasFaqBorrador(BaseModel):
    tono_marca: str = Field(default="Cercano, amable y profesional")
    direccion: str = Field(default="")
    metodos_pago: str = Field(default="")
    politicas: str = Field(default="")
    preguntas_frecuentes: List[PreguntaFrecuenteBorrador] = Field(default_factory=list)


class BorradorNegocioOnboarding(BaseModel):
    id: str = Field(default="", description="Slug unico snake_case del negocio.")
    nombre_negocio: str = Field(default="")
    rubro: str = Field(default="")
    moneda: str = Field(default="COP")
    unidad_cantidad: str = Field(default="servicios")
    telefono_dueno: str = Field(default="")
    whatsapp_negocio: str = Field(default="")
    google_calendar_id: str = Field(default="")
    horario_laboral: HorarioBorrador = Field(default_factory=HorarioBorrador)
    reglas_y_faq: ReglasFaqBorrador = Field(default_factory=ReglasFaqBorrador)
    servicios: List[ServicioBorrador] = Field(default_factory=list)
    promociones: List[PromocionBorrador] = Field(default_factory=list)


class RespuestaOnboarding(BaseModel):
    respuesta_chat: str = Field(
        description="Respuesta natural y breve al dueno pidiendo solo lo que falte o confirmando que el borrador esta listo."
    )
    listo_para_guardar: bool = Field(
        description="True cuando ya hay nombre_negocio, rubro, al menos 1 servicio con precio y horario."
    )
    campos_faltantes: List[str] = Field(default_factory=list)
    borrador_negocio: BorradorNegocioOnboarding


def _declaraciones_herramientas() -> Any:
    """Construye el catalogo de herramientas comerciales para Gemini Tool Calling."""
    if genai_types is None:
        return []
    return [
        genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="consultar_servicios",
                    description=(
                        "Consulta el catalogo oficial de servicios activos, precios exactos, "
                        "duraciones en minutos y promociones vigentes de este negocio."
                    ),
                    parameters_json_schema={"type": "object", "properties": {}},
                ),
                genai_types.FunctionDeclaration(
                    name="calcular_cotizacion",
                    description=(
                        "Calcula con exactitud matematica el valor total de uno o varios servicios "
                        "del catalogo y aplica promociones vigentes si el cliente pide descuento "
                        "o cumple condiciones de combo."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "servicio_clave": {"type": "string"},
                                        "cantidad": {"type": "integer"},
                                    },
                                    "required": ["servicio_clave", "cantidad"],
                                },
                            },
                            "aplicar_promocion": {"type": "boolean"},
                            "diseno_complejo": {"type": "boolean", "description": "True si la clienta pidio diseno elaborado/complejo con costo adicional."},
                            "incluir_domicilio_base": {"type": "boolean", "description": "True si la clienta pidio servicio a domicilio."},
                            "pago_con_tarjeta": {"type": "boolean", "description": "True si la clienta eligio pagar con tarjeta de credito."},
                        },
                        "required": ["items"],
                    },
                ),
                genai_types.FunctionDeclaration(
                    name="consultar_horarios_disponibles",
                    description=(
                        "Consulta los bloques horarios libres reales para una fecha YYYY-MM-DD "
                        "respetando el horario laboral del negocio, Supabase y Google Calendar."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "fecha": {"type": "string", "description": "Fecha en formato YYYY-MM-DD"},
                            "duracion_minutos": {"type": "integer"},
                        },
                        "required": ["fecha"],
                    },
                ),
                genai_types.FunctionDeclaration(
                    name="agendar_cita",
                    description=(
                        "Registra definitivamente una cita tras confirmar horario, servicio, "
                        "nombre completo (nombre y apellido) y telefono del cliente."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "fecha_hora": {"type": "string", "description": "YYYY-MM-DD HH:MM"},
                            "servicio_clave": {"type": "string"},
                            "nombre_cliente": {"type": "string"},
                            "telefono_cliente": {"type": "string"},
                            "cedula_cliente": {"type": "string"},
                            "observaciones": {"type": "string"},
                        },
                        "required": ["fecha_hora", "servicio_clave", "nombre_cliente", "telefono_cliente"],
                    },
                ),
                genai_types.FunctionDeclaration(
                    name="escalar_a_humano",
                    description=(
                        "OBLIGATORIA cuando el cliente: (1) pregunta algo cuya respuesta NO esta "
                        "en los servicios, horarios ni reglas/FAQ de este negocio; (2) presenta "
                        "queja grave, reclamo o insulto; o (3) pide hablar directamente con el dueno o una persona."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "motivo": {"type": "string"},
                            "resumen_para_dueno": {
                                "type": "string",
                                "description": "Resumen claro en 1 o 2 frases para que el dueno sepa exactamente como responder.",
                            },
                            "tipo": {
                                "type": "string",
                                "enum": ["informacion_no_disponible", "reclamo_critico", "solicitud_humana"],
                            },
                        },
                        "required": ["motivo", "resumen_para_dueno", "tipo"],
                    },
                ),
            ]
        )
    ]


def _normalizar_historial_gemini(historial: List[Dict[str, Any]], limite: int = 14) -> List[Dict[str, str]]:
    """
    Garantiza alternancia estricta user/model empezando en 'user' y fusionando turnos consecutivos
    del mismo rol (p. ej. mensajes recibidos mientras el bot estuvo pausado).
    """
    combinados: List[Dict[str, str]] = []
    for turno in historial[-limite:]:
        rol = "user" if turno.get("rol") == "cliente" else "model"
        txt = str(turno.get("texto") or turno.get("contenido") or "").strip()
        if not txt:
            continue
        if combinados and combinados[-1]["role"] == rol:
            combinados[-1]["text"] += "\n" + txt
        else:
            combinados.append({"role": rol, "text": txt})

    while combinados and combinados[0]["role"] != "user":
        combinados.pop(0)
    return combinados


def _construir_instruccion_sistema(negocio: Dict[str, Any]) -> str:
    tz_nombre = str(negocio.get("zona_horaria") or "America/Bogota")
    tz = resolver_zona_horaria_segura(tz_nombre)
    ahora = datetime.now(tz)
    dias_es = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    dia_semana = dias_es[ahora.weekday()]
    reglas = negocio.get("reglas_y_faq") or {}
    horario = negocio.get("horario_laboral") or {}
    instrucciones_extra = str(reglas.get("instrucciones_adicionales") or "").strip()

    base = (
        f"Eres el asistente comercial oficial de '{negocio['nombre_negocio']}' "
        f"(rubro: {negocio['rubro']}, moneda: {negocio['moneda']}).\n"
        f"Fecha y hora actual del negocio: {dia_semana} {ahora.strftime('%Y-%m-%d %H:%M')} ({tz_nombre}).\n"
        f"Tono de marca: {reglas.get('tono_marca') or 'Cercano, amable, claro y profesional.'}\n\n"
        "INFORMACION VERIFICADA DEL NEGOCIO (UNICA fuente autorizada junto con las herramientas):\n"
        f"- Horario laboral: {json.dumps(horario, ensure_ascii=False)}\n"
        f"- Direccion: {reglas.get('direccion') or '(NO REGISTRADA - escalar si preguntan)'}\n"
        f"- Metodos de pago: {reglas.get('metodos_pago') or '(NO REGISTRADOS - escalar si preguntan)'}\n"
        f"- Politicas: {reglas.get('politicas') or '(NO REGISTRADAS - escalar si preguntan)'}\n"
        f"- Preguntas frecuentes registradas: {json.dumps(reglas.get('preguntas_frecuentes', []), ensure_ascii=False)}\n\n"
        "REGLAS ESTRICTAS E INVIOLABLES:\n"
        "1. NUNCA inventes precios, descuentos, horarios disponibles, direcciones, metodos de pago ni politicas.\n"
        "2. Siempre usa `consultar_servicios` para conocer servicios/precios y `calcular_cotizacion` antes de dar totales o descuentos.\n"
        "3. Antes de confirmar disponibilidad o reservar, usa `consultar_horarios_disponibles`. Para llamar a `agendar_cita`, pide primero nombre completo y telefono si aun no los tienes.\n"
        "4. CERO INVENCION: si el cliente pregunta CUALQUIER dato que NO figure en esta ficha ni en las herramientas (o si esta molesto, reclama o pide hablar con una persona/dueno), DEBES llamar inmediatamente a `escalar_a_humano` y avisarle con amabilidad que ya pasaste su consulta al asesor humano.\n"
        "5. Se conciso (maximo 4-5 lineas o lista breve), escribe en espanol natural, sin emojis ni jerga tecnica (nunca menciones 'base de datos', 'herramientas', 'JSON', 'API' ni 'Gemini')."
    )
    if instrucciones_extra:
        base += f"\n\nREGLAS ESPECIFICAS DE ESTE NEGOCIO:\n{instrucciones_extra}"
    return base


class OrquestadorComercialGemini:
    """Orquesta conversaciones multi-negocio sobre Gemini 2.5 Flash con Tool Calling seguro."""

    def __init__(
        self,
        repositorio: RepositorioMultiTenant,
        config: ConfiguracionComercial,
        cliente_genai: Optional[Any] = None,
        fabrica_calendar: Optional[
            Callable[[Dict[str, Any], ConfiguracionComercial], Optional[ClienteGoogleCalendar]]
        ] = None,
    ) -> None:
        self.repo = repositorio
        self.config = config
        self._fabrica_calendar = fabrica_calendar
        if cliente_genai is not None:
            self._cliente = cliente_genai
        else:
            self._cliente = _crear_cliente_genai(config.gemini_api_key)

    @property
    def activo(self) -> bool:
        return self._cliente is not None

    def procesar_mensaje(
        self,
        negocio_id: str,
        id_cliente: str,
        texto: str,
        canal: str = "WhatsApp",
        nombre_cliente: str = "Cliente",
    ) -> Dict[str, Any]:
        negocio = self.repo.obtener_negocio(negocio_id)
        if not negocio:
            raise KeyError(f"Negocio '{negocio_id}' no encontrado.")
        if not negocio.get("activo", True):
            return {
                "status": "inactivo",
                "negocio_id": negocio_id,
                "canal": canal,
                "id_cliente": id_cliente,
                "respuesta": "",
                "responder_al_cliente": False,
                "pausado_humano": True,
                "notificar_dueno": False,
                "telefono_dueno": "",
                "herramientas_usadas": [],
            }

        perfil = self.repo.obtener_o_crear_perfil(
            negocio_id, id_cliente=id_cliente, nombre=nombre_cliente, canal=canal
        )

        # Si este chat esta pausado por Human-in-the-Loop, permanecer en silencio sin gastar tokens
        if not perfil.get("automatizacion_activa", True):
            historial = list(perfil.get("historial_interacciones", []))
            historial.append({"rol": "cliente", "texto": texto})
            perfil["historial_interacciones"] = historial[-30:]
            self.repo.guardar_perfil(negocio_id, perfil)
            return {
                "status": "ok",
                "negocio_id": negocio_id,
                "canal": canal,
                "id_cliente": id_cliente,
                "nombre_cliente": perfil.get("nombre", nombre_cliente),
                "respuesta": "",
                "responder_al_cliente": False,
                "pausado_humano": True,
                "notificar_dueno": False,
                "telefono_dueno": "",
                "mensaje_notificacion_dueno": "",
                "enlace_whatsapp_cliente": "",
                "enlace_panel_dueno": "",
                "herramientas_usadas": [],
            }

        ctx = ContextoHerramientasNegocio(
            repositorio=self.repo,
            negocio=negocio,
            perfil=perfil,
            canal=canal,
            config=self.config,
            fabrica_calendar=self._fabrica_calendar,
        )

        historial = list(perfil.get("historial_interacciones", []))
        historial.append({"rol": "cliente", "texto": texto})
        perfil["historial_interacciones"] = historial[-30:]

        if not self.activo:
            ctx.escalar_a_humano(
                motivo="Motor Gemini sin GEMINI_API_KEY configurada",
                resumen_para_dueno=f"El cliente escribio: '{texto[:140]}'. Falta configurar GEMINI_API_KEY en el servidor.",
                tipo="informacion_no_disponible",
            )
            respuesta_texto = (
                "En este momento estoy transfiriendo tu mensaje con nuestro asesor encargado "
                "para atenderte directamente por este mismo chat."
            )
        else:
            respuesta_texto = self._ejecutar_bucle_tool_calling(negocio, historial, ctx, texto)

        historial.append({"rol": "asistente", "texto": respuesta_texto})
        perfil["historial_interacciones"] = historial[-30:]
        self.repo.guardar_perfil(negocio_id, perfil)

        esc = ctx.escalamiento_disparado or {}
        notificar = bool(esc.get("escalado", False))
        return {
            "status": "ok",
            "negocio_id": negocio_id,
            "canal": canal,
            "id_cliente": id_cliente,
            "nombre_cliente": perfil.get("nombre", nombre_cliente),
            "respuesta": respuesta_texto,
            "responder_al_cliente": True,
            "pausado_humano": bool(esc.get("pausado_humano", False)),
            "notificar_dueno": notificar,
            "telefono_dueno": str(esc.get("telefono_dueno") or "") if notificar else "",
            "mensaje_notificacion_dueno": str(esc.get("mensaje_notificacion_dueno") or "") if notificar else "",
            "enlace_whatsapp_cliente": str(esc.get("enlace_whatsapp_cliente") or "") if notificar else "",
            "enlace_panel_dueno": str(esc.get("enlace_panel_dueno") or "") if notificar else "",
            "cita_confirmada": ctx.cita_confirmada,
            "herramientas_usadas": [h["herramienta"] for h in ctx.herramientas_ejecutadas],
        }

    def _despachar_herramienta(
        self, ctx: ContextoHerramientasNegocio, nombre: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        mapa: Dict[str, Callable[..., Dict[str, Any]]] = {
            "consultar_servicios": ctx.consultar_servicios,
            "calcular_cotizacion": ctx.calcular_cotizacion,
            "consultar_horarios_disponibles": ctx.consultar_horarios_disponibles,
            "agendar_cita": ctx.agendar_cita,
            "escalar_a_humano": ctx.escalar_a_humano,
        }
        fn = mapa.get(nombre)
        if fn is None:
            return {"error": f"Herramienta desconocida: {nombre}"}
        try:
            return fn(**(args or {}))
        except Exception as exc:
            logger.warning("Error ejecutando herramienta %s: %s", nombre, exc)
            return {"error": "No se pudo completar la operacion solicitada en este momento."}

    def _ejecutar_bucle_tool_calling(
        self,
        negocio: Dict[str, Any],
        historial: List[Dict[str, Any]],
        ctx: ContextoHerramientasNegocio,
        ultimo_texto: str,
    ) -> str:
        turnos = _normalizar_historial_gemini(historial, limite=14)
        contenidos: List[Any] = []
        for t in turnos:
            if genai_types is not None:
                contenidos.append(
                    genai_types.Content(
                        role=t["role"],
                        parts=[genai_types.Part.from_text(text=t["text"])],
                    )
                )
            else:
                contenidos.append({"role": t["role"], "parts": [{"text": t["text"]}]})

        config_gen = (
            genai_types.GenerateContentConfig(
                system_instruction=_construir_instruccion_sistema(negocio),
                tools=_declaraciones_herramientas(),
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.25,
                max_output_tokens=600,
            )
            if genai_types is not None
            else {}
        )

        for _ in range(6):
            try:
                resp = self._cliente.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contenidos,
                    config=config_gen,
                )
            except Exception as exc:
                logger.error("Fallo invocando Gemini para negocio '%s': %s", negocio.get("id"), exc)
                ctx.escalar_a_humano(
                    motivo="Interrupcion temporal del motor de IA",
                    resumen_para_dueno=f"El cliente escribio: '{ultimo_texto[:140]}'. El asistente no pudo responder temporalmente.",
                    tipo="informacion_no_disponible",
                )
                return (
                    "Disculpa la demora, en este momento estoy pasando tu mensaje a nuestro asesor "
                    "para que te responda directamente por este mismo chat."
                )

            calls = getattr(resp, "function_calls", None) or []
            if not calls:
                texto_final = str(getattr(resp, "text", "") or "").strip()
                if texto_final:
                    return texto_final
                break

            candidatos = getattr(resp, "candidates", None) or []
            if candidatos and getattr(candidatos[0], "content", None) is not None:
                contenidos.append(candidatos[0].content)

            partes_respuesta: List[Any] = []
            for call in calls:
                nombre_fn = str(getattr(call, "name", "") or "")
                args_raw = getattr(call, "args", None) or {}
                args_dict = dict(args_raw) if not isinstance(args_raw, dict) else args_raw
                resultado_tool = self._despachar_herramienta(ctx, nombre_fn, args_dict)
                if genai_types is not None:
                    partes_respuesta.append(
                        genai_types.Part.from_function_response(
                            name=nombre_fn,
                            response={"resultado": resultado_tool},
                        )
                    )
                else:
                    partes_respuesta.append(
                        {"function_response": {"name": nombre_fn, "response": {"resultado": resultado_tool}}}
                    )
            if genai_types is not None:
                contenidos.append(genai_types.Content(role="user", parts=partes_respuesta))
            else:
                contenidos.append({"role": "user", "parts": partes_respuesta})

        if ctx.escalamiento_disparado:
            return (
                "No cuento con esa informacion verificada en este momento, asi que ya escale "
                "tu consulta con nuestro asesor encargado para que te responda por este mismo chat."
            )
        return "Con gusto te ayudo. Cuentame que servicio o fecha te interesa consultar."


class AsistenteOnboardingGemini:
    """
    Asistente conversacional de alta de negocios con Structured Outputs.
    Entrevista al dueno o lee un menu/lista de precios pegada y produce un borrador
    listo para revisar y guardar visualmente en el panel PWA.
    """

    def __init__(self, config: ConfiguracionComercial, cliente_genai: Optional[Any] = None) -> None:
        self.config = config
        if cliente_genai is not None:
            self._cliente = cliente_genai
        else:
            self._cliente = _crear_cliente_genai(config.gemini_api_key)

    def procesar_turno(
        self,
        historial: List[Dict[str, str]],
        mensaje_usuario: str,
        borrador_actual: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        borrador_base = borrador_actual or BorradorNegocioOnboarding().model_dump()
        if self._cliente is None:
            return {
                "respuesta_chat": (
                    "GEMINI_API_KEY no esta configurada en el servidor. Puedes llenar el formulario "
                    "manual de la derecha o configurar GEMINI_API_KEY en .env para activar el asistente IA."
                ),
                "listo_para_guardar": False,
                "campos_faltantes": ["GEMINI_API_KEY"],
                "borrador_negocio": borrador_base,
            }

        instruccion = (
            "Eres un consultor de configuracion de negocios para una plataforma de asistentes de WhatsApp. "
            "Tu tarea es entrevistar al usuario (o extraer datos de un menu/lista de precios que pegue) "
            "para llenar el borrador JSON de su negocio: nombre_negocio, rubro, moneda (default COP), "
            "unidad_cantidad, telefono_dueno, whatsapp_negocio, horario_laboral (dias 1=Lunes..7=Domingo, "
            "inicio HH:MM, fin HH:MM, duracion_min), reglas_y_faq (tono_marca, direccion, metodos_pago, "
            "politicas, preguntas_frecuentes), servicios (clave snake_case, nombre, descripcion, "
            "tarifa_unitaria numerica, duracion_minutos) y promociones opcionales.\n"
            "Conserva SIEMPRE todos los datos que ya estaban en el borrador_actual y solo agrega o actualiza "
            "con la nueva informacion del usuario. Si faltan datos esenciales (nombre, rubro, servicios con precio "
            "o horario), pidelos de forma breve y ordenada en respuesta_chat."
        )

        contexto_txt = (
            f"Borrador actual:\n{json.dumps(borrador_actual or {}, ensure_ascii=False)}\n\n"
            f"Historial:\n{json.dumps(historial[-10:], ensure_ascii=False)}\n\n"
            f"Nuevo mensaje del usuario:\n{mensaje_usuario}"
        )

        config_gen = (
            genai_types.GenerateContentConfig(
                system_instruction=instruccion,
                response_mime_type="application/json",
                response_schema=RespuestaOnboarding,
                temperature=0.15,
            )
            if genai_types is not None
            else {}
        )

        try:
            resp = self._cliente.models.generate_content(
                model=self.config.gemini_model,
                contents=contexto_txt,
                config=config_gen,
            )
        except Exception as exc:
            logger.error("Error invocando Gemini en onboarding: %s", exc)
            return {
                "respuesta_chat": (
                    "El asistente de configuracion no pudo responder en este momento. "
                    "Puedes reintentar o editar directamente el formulario de la derecha."
                ),
                "listo_para_guardar": False,
                "campos_faltantes": [],
                "borrador_negocio": borrador_base,
            }

        raw = str(getattr(resp, "text", "") or "{}").strip()
        try:
            validado = RespuestaOnboarding.model_validate_json(raw)
            return validado.model_dump()
        except Exception:
            try:
                datos = json.loads(raw)
                return RespuestaOnboarding.model_validate(datos).model_dump()
            except Exception:
                return {
                    "respuesta_chat": "Recibi tus datos. Podrias confirmarme el nombre del negocio, servicios con precios y horario?",
                    "listo_para_guardar": False,
                    "campos_faltantes": ["nombre_negocio", "servicios"],
                    "borrador_negocio": borrador_base,
                }
