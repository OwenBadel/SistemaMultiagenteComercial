"""
Suite de pruebas TDD para la plataforma comercial multi-negocio (rama comercial):
- Aislamiento estricto entre negocios (Fotografia, Belleza y Pizzeria dinamica)
- Seguridad AppSec: anti-hijacking de canales, RBAC, fail-closed webhooks, HMAC Meta, rate-limit y revocacion de PIN
- Motor Gemini 2.5 Flash con Tool Calling, resiliencia ante errores e idempotencia de escalamiento
- Sincronizacion real con Google Calendar (duracion del servicio y persistencia en BD)
- Paridad de RepositorioComercialSupabase con mock HTTPX
- Onboarding conversacional estructurado con Gemini y Panel PWA
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from src.comercial.configuracion import ConfiguracionComercial
from src.comercial.herramientas_negocio import ContextoHerramientasNegocio
from src.comercial.motor_gemini import (
    AsistenteOnboardingGemini,
    OrquestadorComercialGemini,
    _normalizar_historial_gemini,
)
from src.comercial.notificador_whatsapp import enviar_mensaje_whatsapp_cloud
from src.comercial.repositorio_multitenant import (
    ConflictoAgendaError,
    ConflictoCanalError,
    RepositorioComercialSQLite,
    RepositorioComercialSupabase,
)
from src.comercial.servidor_comercial import crear_app_comercial


def _proximo_dia_habil(dias_permitidos: List[int]) -> date:
    d = date.today() + timedelta(days=2)
    while d.isoweekday() not in dias_permitidos:
        d += timedelta(days=1)
    return d


def _proximo_lunes() -> date:
    d = date.today() + timedelta(days=2)
    while d.isoweekday() != 1:
        d += timedelta(days=1)
    return d


def _firmar_meta(cuerpo: bytes, secreto: str) -> str:
    return "sha256=" + hmac.new(secreto.encode("utf-8"), cuerpo, hashlib.sha256).hexdigest()


class _ClienteGeminiFake:
    """Simulador determinista de `google.genai.Client` para pruebas TDD offline."""

    def __init__(self, respuestas: List[Any], fallar_con: Exception | None = None) -> None:
        self._respuestas = list(respuestas)
        self._fallar_con = fallar_con
        self.llamadas: List[Dict[str, Any]] = []
        self.models = self

    def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
        self.llamadas.append({"model": model, "contents": contents, "config": config})
        if self._fallar_con is not None:
            raise self._fallar_con
        if not self._respuestas:
            return SimpleNamespace(text="Respuesta de respaldo.", function_calls=[], candidates=[])
        return self._respuestas.pop(0)


def _resp_tool_call(nombre: str, args: Dict[str, Any]) -> Any:
    call = SimpleNamespace(name=nombre, args=args)
    return SimpleNamespace(
        text="",
        function_calls=[call],
        candidates=[
            SimpleNamespace(
                content={"role": "model", "parts": [{"function_call": {"name": nombre, "args": args}}]}
            )
        ],
    )


def _resp_texto(texto: str) -> Any:
    return SimpleNamespace(text=texto, function_calls=[], candidates=[])


def _config_prueba(**overrides: Any) -> ConfiguracionComercial:
    base = dict(
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        sqlite_db_path=":memory:",
        admin_access_key="clave_super_test",
        session_secret="secreto_sesion_test_largo_y_seguro",
        botpress_api_key="bp_key_test_123",
        whatsapp_verify_token="verify_meta_test",
        whatsapp_app_secret="meta_secret_test_456",
        public_base_url="https://panel.ejemplo.com",
    )
    base.update(overrides)
    return ConfiguracionComercial(**base)


def test_aislamiento_estricto_entre_fotografia_y_belleza() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    negocios = {n["id"]: n for n in repo.listar_negocios()}
    assert "estudio_fotografia" in negocios
    assert "salon_belleza" in negocios

    # Mismo cliente escribe a dos negocios diferentes -> perfiles independientes
    p_foto = repo.obtener_o_crear_perfil("estudio_fotografia", "573001112233", "Ana Ruiz")
    repo.obtener_o_crear_perfil("salon_belleza", "573001112233", "Ana Ruiz")
    p_foto["automatizacion_activa"] = False
    repo.guardar_perfil("estudio_fotografia", p_foto)

    assert repo.obtener_o_crear_perfil("estudio_fotografia", "573001112233")["automatizacion_activa"] is False
    assert repo.obtener_o_crear_perfil("salon_belleza", "573001112233")["automatizacion_activa"] is True

    # Misma fecha y hora en dos negocios diferentes NO choca; en el mismo negocio SI choca
    dia = _proximo_dia_habil([2, 3, 4, 5, 6])
    ini = f"{dia.isoformat()}T10:00:00-05:00"
    fin = f"{dia.isoformat()}T11:00:00-05:00"
    repo.registrar_agendamiento(
        "estudio_fotografia",
        {"id_cliente": "573001112233", "nombre_cliente": "Ana Ruiz", "canal": "WhatsApp", "servicio": "Retrato", "fecha_inicio": ini, "fecha_fin": fin},
    )
    repo.registrar_agendamiento(
        "salon_belleza",
        {"id_cliente": "573001112233", "nombre_cliente": "Ana Ruiz", "canal": "WhatsApp", "servicio": "Manicure", "fecha_inicio": ini, "fecha_fin": fin},
    )
    with pytest.raises(ConflictoAgendaError):
        repo.registrar_agendamiento(
            "salon_belleza",
            {"id_cliente": "573009998877", "nombre_cliente": "Sara Diaz", "canal": "WhatsApp", "servicio": "Pedicure", "fecha_inicio": ini, "fecha_fin": fin},
        )


def test_herramientas_calculo_exacto_horarios_por_negocio_y_agenda() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    cfg = _config_prueba()
    neg_belleza = repo.obtener_negocio("salon_belleza")
    assert neg_belleza is not None
    perfil = repo.obtener_o_crear_perfil("salon_belleza", "573005556677", "Camila Torres")
    ctx = ContextoHerramientasNegocio(repo, neg_belleza, perfil, "WhatsApp", cfg)

    # 1. Catalogo aislado de belleza
    cat = ctx.consultar_servicios()
    claves = {s["clave"] for s in cat["servicios"]}
    assert "manicure_semipermanente" in claves
    assert "cobertura_evento" not in claves

    # 2. Cotizacion exacta con promocion combo (2 servicios -> 10% descuento)
    cot = ctx.calcular_cotizacion(
        [
            {"servicio_clave": "manicure_semipermanente", "cantidad": 1},
            {"servicio_clave": "pedicure_spa", "cantidad": 1},
        ],
        aplicar_promocion=True,
    )
    assert cot["subtotal"] == 100000.0
    assert cot["descuento_porciento"] == 10.0
    assert cot["total"] == 90000.0

    # 3. Horario laboral por negocio: salon_belleza NO abre los lunes (dias=[2,3,4,5,6])
    lunes = _proximo_lunes()
    disp_lunes = ctx.consultar_horarios_disponibles(lunes.isoformat(), 60)
    assert disp_lunes["dia_laboral"] is False
    assert disp_lunes["bloques_disponibles"] == []

    # 4. Dia habil: devuelve bloques y permite agendar
    dia_ok = _proximo_dia_habil([2, 3, 4, 5, 6])
    disp_ok = ctx.consultar_horarios_disponibles(dia_ok.isoformat(), 60)
    assert disp_ok["dia_laboral"] is True
    assert len(disp_ok["bloques_disponibles"]) >= 4

    cita = ctx.agendar_cita(
        fecha_hora=f"{dia_ok.isoformat()} 09:00",
        servicio_clave="manicure_semipermanente",
        nombre_cliente="Camila Torres",
        telefono_cliente="573005556677",
    )
    assert cita["confirmada"] is True
    assert cita["servicio"] == "Manicure Semipermanente"

    # 5. Tras cita de 09:00-10:00, un servicio de 45 min debe ofrecer 10:00 inmediatamente sin saltarlo
    disp_45 = ctx.consultar_horarios_disponibles(dia_ok.isoformat(), 45)
    assert f"{dia_ok.isoformat()} 10:00" in disp_45["bloques_disponibles"]

    # 6. Intento de agendar en el mismo bloque ocupado devuelve alternativas
    cita_dup = ctx.agendar_cita(
        fecha_hora=f"{dia_ok.isoformat()} 09:00",
        servicio_clave="pedicure_spa",
        nombre_cliente="Lucia Perez",
        telefono_cliente="573001239999",
    )
    assert cita_dup["confirmada"] is False
    assert len(cita_dup["bloques_alternativos"]) >= 1


def test_google_calendar_sync_real_y_resiliencia_gemini() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    cfg = _config_prueba()
    neg = repo.obtener_negocio("salon_belleza")
    assert neg is not None
    perfil = repo.obtener_o_crear_perfil("salon_belleza", "573004445566", "Paula Mendoza")

    eventos_creados: List[Dict[str, Any]] = []

    class _FakeCalendar:
        def __init__(self) -> None:
            self.duracion_minutos = 60

        def _consultar_freebusy(self, ini: Any, fin: Any) -> List[Any]:
            return []

        def crear_evento(self, **kwargs: Any) -> Dict[str, Any]:
            eventos_creados.append({"duracion": self.duracion_minutos, **kwargs})
            return {
                "fecha_hora": kwargs["fecha_hora"],
                "estado": "OCUPADO",
                "cliente": kwargs["cliente"],
                "evento_google_id": "evt_gcal_999",
                "html_link": "https://calendar.google.com/event?eid=999",
            }

    fake_cal = _FakeCalendar()
    ctx = ContextoHerramientasNegocio(
        repo, neg, perfil, "WhatsApp", cfg, fabrica_calendar=lambda _n, _c: fake_cal
    )

    dia_ok = _proximo_dia_habil([2, 3, 4, 5, 6])
    cita = ctx.agendar_cita(
        fecha_hora=f"{dia_ok.isoformat()} 10:00",
        servicio_clave="balayage_iluminacion",
        nombre_cliente="Paula Mendoza",
        telefono_cliente="573004445566",
    )
    assert cita["confirmada"] is True
    assert cita["duracion_minutos"] == 180
    assert eventos_creados[0]["duracion"] == 180
    assert cita["google_calendar_estado"] == "sincronizado"
    assert cita["google_event_id"] == "evt_gcal_999"
    assert cita["google_event_link"] == "https://calendar.google.com/event?eid=999"

    # Verificar que los datos de Google Calendar quedaron persistidos en BD
    en_bd = repo.obtener_agendamiento("salon_belleza", str(cita["agendamiento_id"]))
    assert en_bd is not None
    assert en_bd["registrado_en"] == "supabase_y_google_calendar"
    assert en_bd["google_calendar_estado"] == "sincronizado"
    assert en_bd["google_event_id"] == "evt_gcal_999"

    # Idempotencia de escalar_a_humano en el mismo turno (no crea alertas duplicadas)
    ctx.escalar_a_humano("Motivo 1", "Resumen 1")
    ctx.escalar_a_humano("Motivo 2", "Resumen 2")
    assert len(repo.listar_alertas("salon_belleza", solo_abiertas=True)) == 1

    # Normalizacion de historial Gemini: fusiona turnos consecutivos y descarta 'model' inicial
    norm = _normalizar_historial_gemini(
        [
            {"rol": "asistente", "texto": "Hola"},
            {"rol": "cliente", "texto": "Msg 1"},
            {"rol": "cliente", "texto": "Msg 2"},
            {"rol": "asistente", "texto": "Resp 1"},
            {"rol": "cliente", "texto": "Msg 3"},
        ]
    )
    assert [t["role"] for t in norm] == ["user", "model", "user"]
    assert "Msg 1\nMsg 2" == norm[0]["text"]

    # Resiliencia ante error de red / 429 en Gemini: escala sin lanzar 500
    gemini_roto = _ClienteGeminiFake([], fallar_con=RuntimeError("429 RESOURCE_EXHAUSTED"))
    orq = OrquestadorComercialGemini(repo, cfg, cliente_genai=gemini_roto)
    res_err = orq.procesar_mensaje("estudio_fotografia", "573001010101", "Hola quiero cotizar")
    assert res_err["pausado_humano"] is True
    assert res_err["notificar_dueno"] is True
    assert "asesor" in res_err["respuesta"].lower()


def test_flujo_gemini_tool_calling_y_escalamiento_humano_silencioso() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    cfg = _config_prueba()
    repo.guardar_negocio({"id": "salon_belleza", "telefono_dueno": "573112223344"})

    gemini_fake = _ClienteGeminiFake(
        [
            _resp_tool_call("consultar_servicios", {}),
            _resp_texto("El Manicure Semipermanente cuesta $45,000 COP y dura 60 minutos."),
            _resp_tool_call(
                "escalar_a_humano",
                {
                    "motivo": "Consulta sobre tintura hipoalergenica medica no documentada",
                    "resumen_para_dueno": "Camila pregunta si manejan tinturas dermatologicas sin amoniaco.",
                    "tipo": "informacion_no_disponible",
                },
            ),
            _resp_texto(
                "No tengo ese detalle verificado en este momento, asi que ya pase tu consulta a nuestra asesora."
            ),
        ]
    )
    orq = OrquestadorComercialGemini(repo, cfg, cliente_genai=gemini_fake)

    r1 = orq.procesar_mensaje("salon_belleza", "573008887766", "Cuanto vale el manicure?", nombre_cliente="Camila")
    assert "45,000" in r1["respuesta"]
    assert r1["pausado_humano"] is False
    assert r1["telefono_dueno"] == ""  # No expone telefono del dueno si no hay escalamiento
    assert "consultar_servicios" in r1["herramientas_usadas"]

    r2 = orq.procesar_mensaje("salon_belleza", "573008887766", "Usan tintura hipoalergenica medica?", nombre_cliente="Camila")
    assert r2["pausado_humano"] is True
    assert r2["notificar_dueno"] is True
    assert r2["telefono_dueno"] == "573112223344"
    assert "https://wa.me/573008887766" in r2["mensaje_notificacion_dueno"]
    assert "https://panel.ejemplo.com/?negocio=salon_belleza#alertas" in r2["mensaje_notificacion_dueno"]

    # Turno 3 mientras esta pausado: silencio total SIN volver a invocar a Gemini ni filtrar telefono_dueno
    llamadas_antes = len(gemini_fake.llamadas)
    r3 = orq.procesar_mensaje("salon_belleza", "573008887766", "Hola? siguen ahi?", nombre_cliente="Camila")
    assert r3["responder_al_cliente"] is False
    assert r3["respuesta"] == ""
    assert r3["telefono_dueno"] == ""
    assert len(gemini_fake.llamadas) == llamadas_antes


def test_api_roles_onboarding_gemini_y_webhook_meta_whatsapp() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    cfg = _config_prueba()
    repo.guardar_negocio(
        {
            "id": "estudio_fotografia",
            "whatsapp_phone_number_id": "100200300401",
            "whatsapp_access_token": "tok_foto_secreto",
            "botpress_bot_id": "bp_foto_01",
            "telefono_dueno": "573100001111",
            "pin_dueno": "4321",
        }
    )
    repo.guardar_negocio(
        {
            "id": "salon_belleza",
            "whatsapp_phone_number_id": "100200300402",
            "whatsapp_access_token": "tok_belleza_secreto",
            "botpress_bot_id": "bp_belleza_02",
            "telefono_dueno": "573200002222",
            "pin_dueno": "9876",
        }
    )

    envios_wa: List[Dict[str, Any]] = []

    def _fake_wa(**kwargs: Any) -> Dict[str, Any]:
        envios_wa.append(kwargs)
        return {"enviado": True}

    gemini_chat = _ClienteGeminiFake(
        [
            _resp_tool_call(
                "escalar_a_humano",
                {
                    "motivo": "Pide hablar con la duena del salon",
                    "resumen_para_dueno": "Valeria quiere coordinar un paquete de novia personalizado.",
                    "tipo": "solicitud_humana",
                },
            ),
            _resp_texto("Con gusto, ya le avise a la encargada del salon para que te escriba por aqui."),
            _resp_texto("Hola, bienvenido al estudio de fotografia."),
        ]
    )
    gemini_onb = _ClienteGeminiFake(
        [
            _resp_texto(
                json.dumps(
                    {
                        "respuesta_chat": "Listo, arme el borrador de Pizzeria Don Luigi con sus 2 pizzas y horario.",
                        "listo_para_guardar": True,
                        "campos_faltantes": [],
                        "borrador_negocio": {
                            "id": "pizzeria_don_luigi",
                            "nombre_negocio": "Pizzeria Don Luigi",
                            "rubro": "Pizzeria Artesanal y Domicilios",
                            "moneda": "COP",
                            "unidad_cantidad": "pizzas",
                            "telefono_dueno": "573155550000",
                            "horario_laboral": {"dias": [1, 2, 3, 4, 5, 6, 7], "inicio": "17:00", "fin": "23:00", "duracion_min": 30},
                            "reglas_y_faq": {
                                "tono_marca": "Alegre y rapido",
                                "direccion": "Calle 10 # 5-20",
                                "metodos_pago": "Nequi y efectivo",
                                "politicas": "Domicilios en 40 minutos",
                                "preguntas_frecuentes": [],
                            },
                            "servicios": [
                                {"clave": "pizza_margarita", "nombre": "Pizza Margarita", "descripcion": "Masa madre y albahaca", "tarifa_unitaria": 32000, "duracion_minutos": 30},
                                {"clave": "pizza_pepperoni", "nombre": "Pizza Pepperoni", "descripcion": "Extra queso y pepperoni", "tarifa_unitaria": 36000, "duracion_minutos": 30},
                            ],
                            "promociones": [],
                        },
                    }
                )
            )
        ]
    )

    app = crear_app_comercial(
        config=cfg,
        repositorio=repo,
        orquestador=OrquestadorComercialGemini(repo, cfg, cliente_genai=gemini_chat),
        onboarding=AsistenteOnboardingGemini(cfg, cliente_genai=gemini_onb),
        enviador_whatsapp=_fake_wa,
    )
    client = TestClient(app)

    # 1. Login dueno de Fotografia: accede a Fotografia pero recibe 403 en Belleza, Admin y Onboarding
    r_login_foto = client.post(
        "/api/auth/login",
        json={"rol": "dueno", "negocio_id": "estudio_fotografia", "clave": "4321"},
    )
    assert r_login_foto.status_code == 200
    hdr_foto = {"Authorization": f"Bearer {r_login_foto.json()['token']}"}

    res_foto = client.get("/api/negocios/estudio_fotografia/resumen", headers=hdr_foto)
    assert res_foto.status_code == 200
    assert "pin_dueno_hash" not in res_foto.json()["negocio"]
    assert "whatsapp_access_token" not in res_foto.json()["negocio"]

    assert client.get("/api/negocios/salon_belleza/resumen", headers=hdr_foto).status_code == 403
    assert client.get("/api/admin/negocios", headers=hdr_foto).status_code == 403
    assert client.post("/api/onboarding/chat", headers=hdr_foto, json={"mensaje": "hola"}).status_code == 403

    # Anti-hijacking: dueno NO puede modificar whatsapp_phone_number_id ni margenes (403)
    assert (
        client.put(
            "/api/negocios/estudio_fotografia/configuracion",
            headers=hdr_foto,
            json={"whatsapp_phone_number_id": "100200300402"},
        ).status_code
        == 403
    )

    # Actualizacion parcial de dueno NO borra botpress_bot_id ni margenes
    r_upd = client.put(
        "/api/negocios/estudio_fotografia/configuracion",
        headers=hdr_foto,
        json={"telefono_dueno": "573109998877", "pin_dueno": "7788"},
    )
    assert r_upd.status_code == 200
    assert r_upd.json()["negocio"]["botpress_bot_id"] == "bp_foto_01"
    assert r_upd.json()["negocio"]["margen_base"] == 0.40
    assert "nuevo_token" in r_upd.json()
    # Token anterior del dueno queda revocado inmediatamente tras cambio de PIN
    assert client.get("/api/negocios/estudio_fotografia/resumen", headers=hdr_foto).status_code == 401
    hdr_foto_nuevo = {"Authorization": f"Bearer {r_upd.json()['nuevo_token']}"}
    assert client.get("/api/negocios/estudio_fotografia/resumen", headers=hdr_foto_nuevo).status_code == 200

    # 2. Login Super-Admin -> Onboarding Gemini -> Crear Pizzeria -> Colision de canal (409)
    r_login_admin = client.post(
        "/api/auth/login",
        json={"rol": "superadmin", "clave": "clave_super_test"},
    )
    assert r_login_admin.status_code == 200
    hdr_admin = {"Authorization": f"Bearer {r_login_admin.json()['token']}"}

    r_onb = client.post(
        "/api/onboarding/chat",
        headers=hdr_admin,
        json={"historial": [], "mensaje": "Pizzeria Don Luigi, Pizza Margarita 32mil y Pepperoni 36mil, 5pm a 11pm"},
    )
    assert r_onb.status_code == 200
    borrador = r_onb.json()["borrador_negocio"]
    assert borrador["nombre_negocio"] == "Pizzeria Don Luigi"

    # Colision de whatsapp_phone_number_id debe retornar 409
    assert (
        client.post(
            "/api/admin/negocios",
            headers=hdr_admin,
            json={**borrador, "whatsapp_phone_number_id": "100200300402", "pin_dueno": "5555"},
        ).status_code
        == 409
    )

    r_crear = client.post("/api/admin/negocios", headers=hdr_admin, json={**borrador, "pin_dueno": "5555"})
    assert r_crear.status_code == 200
    assert r_crear.json()["negocio"]["id"] == "pizzeria_don_luigi"
    assert len(r_crear.json()["servicios"]) == 2

    # 3. Verificacion Meta WhatsApp Webhook GET + firma HMAC obligatoria en POST
    r_verify = client.get(
        "/api/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify_meta_test", "hub.challenge": "998877"},
    )
    assert r_verify.status_code == 200
    assert r_verify.text == "998877"

    payload_meta = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": "100200300402", "display_phone_number": "573001119999"},
                            "contacts": [{"wa_id": "573007776655", "profile": {"name": "Valeria Castro"}}],
                            "messages": [{"from": "573007776655", "type": "text", "text": {"body": "Quiero hablar con la duena para un paquete de novia"}}],
                        }
                    }
                ]
            }
        ]
    }
    raw_meta = json.dumps(payload_meta).encode("utf-8")
    # Sin firma valida -> 401
    assert client.post("/api/whatsapp/webhook", content=raw_meta, headers={"Content-Type": "application/json"}).status_code == 401

    firma_valida = _firmar_meta(raw_meta, cfg.whatsapp_app_secret)
    r_meta = client.post(
        "/api/whatsapp/webhook",
        content=raw_meta,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": firma_valida},
    )
    assert r_meta.status_code == 200
    assert r_meta.json()["procesados"] == 1
    assert r_meta.json()["detalle"][0]["negocio_id"] == "salon_belleza"
    assert len(envios_wa) == 2
    assert {e["destinatario"] for e in envios_wa} == {"573007776655", "573200002222"}

    # Botpress webhook: rechaza sin key (401), acepta X-Botpress-Key (200)
    assert client.post("/api/webhook/estudio_fotografia", json={"id_cliente": "c1", "texto": "hola"}).status_code == 401
    r_bp = client.post(
        "/api/webhook/estudio_fotografia",
        headers={"X-Botpress-Key": cfg.botpress_api_key},
        json={"id_cliente": "c1", "texto": "hola"},
    )
    assert r_bp.status_code == 200
    assert r_bp.json()["telefono_dueno"] == ""

    # 4. Dueno de Belleza ve el chat pausado, 404 si intenta reactivar cliente inexistente, y reactiva el real
    r_login_bell = client.post(
        "/api/auth/login",
        json={"rol": "dueno", "negocio_id": "salon_belleza", "clave": "9876"},
    )
    hdr_bell = {"Authorization": f"Bearer {r_login_bell.json()['token']}"}
    res_bell = client.get("/api/negocios/salon_belleza/resumen", headers=hdr_bell).json()
    assert res_bell["kpis"]["chats_pausados"] == 1
    assert res_bell["chats_pausados"][0]["id_cliente"] == "573007776655"

    assert client.post("/api/negocios/salon_belleza/chats/no_existe/reactivar", headers=hdr_bell).status_code == 404
    assert client.post("/api/negocios/salon_belleza/chats/573007776655/reactivar", headers=hdr_bell).status_code == 200
    assert client.get("/api/negocios/salon_belleza/resumen", headers=hdr_bell).json()["kpis"]["chats_pausados"] == 0

    # 5. Rate-limit login (6 intentos fallidos -> 7mo devuelve 429)
    for _ in range(6):
        assert (
            client.post(
                "/api/auth/login",
                json={"rol": "dueno", "negocio_id": "salon_belleza", "clave": "0000"},
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/api/auth/login",
            json={"rol": "dueno", "negocio_id": "salon_belleza", "clave": "0000"},
        ).status_code
        == 429
    )

    # 6. PWA HTML, headers de seguridad, manifest y service worker con no-cache
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert r_home.headers.get("x-content-type-options") == "nosniff"
    assert r_home.headers.get("x-frame-options") == "DENY"
    assert "script-src 'self'" in r_home.headers.get("content-security-policy", "")
    assert client.get("/manifest.json").status_code == 200
    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert r_sw.headers.get("cache-control") == "no-cache"


def test_webhooks_fail_closed_y_validacion_phone_number_id() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    cfg_sin_llaves = _config_prueba(botpress_api_key="", whatsapp_app_secret="", allow_insecure_webhooks=False)
    app = crear_app_comercial(config=cfg_sin_llaves, repositorio=repo)
    client = TestClient(app)

    assert client.post("/api/webhook/estudio_fotografia", json={"id_cliente": "c1", "texto": "hola"}).status_code == 503
    assert client.post("/api/whatsapp/webhook", json={"entry": []}).status_code == 503

    # Validacion anti path-injection en enviar_mensaje_whatsapp_cloud
    res_bad = enviar_mensaje_whatsapp_cloud("../evil/path", "tok", "573001112233", "hola")
    assert res_bad["enviado"] is False
    assert res_bad["motivo"] == "phone_number_id_invalido"


def test_paridad_repositorio_supabase_con_mock_httpx() -> None:
    peticiones: List[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cuerpo = json.loads(request.content.decode("utf-8")) if request.content else None
        ruta = request.url.path.split("/rest/v1/")[-1]
        query = request.url.query.decode("utf-8")
        peticiones.append((request.method, f"{ruta}?{query}", cuerpo))

        if ruta == "servicios" and request.method == "GET":
            if "activo=eq.true" in query:
                return httpx.Response(400, json={"message": "column activo does not exist"})
            return httpx.Response(
                200,
                json=[
                    {"negocio_id": "n1", "clave": "s1", "servicio_id_num": 1, "nombre": "S1", "tarifa_unitaria": 1000, "activo": True},
                    {"negocio_id": "n1", "clave": "s2", "servicio_id_num": 2, "nombre": "S2", "tarifa_unitaria": 2000, "activo": False},
                ],
            )
        if ruta == "servicios" and request.method == "POST":
            return httpx.Response(201, json=[cuerpo])
        if ruta == "agendamientos" and request.method == "POST":
            return httpx.Response(409, json={"code": "23P01", "message": "conflicting key value"})
        if ruta == "perfiles_clientes" and request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "negocio_id": "n1",
                        "id_cliente": "573001112233",
                        "nombre": "Cliente",
                        "canal": "WhatsApp",
                        "historial_interacciones": [],
                        "parametros_parciales": {},
                        "automatizacion_activa": True,
                    }
                ],
            )
        if ruta == "perfiles_clientes" and request.method == "POST":
            return httpx.Response(200, json=[cuerpo])
        return httpx.Response(200, json=[])

    cliente_http = httpx.Client(transport=httpx.MockTransport(handler))
    repo = RepositorioComercialSupabase("https://demo.supabase.co", "sb_secret_test", http_client=cliente_http)

    # 1. Fallback de listar_servicios(solo_activos=True) filtra inactivos en Python
    activos = repo.listar_servicios("n1", solo_activos=True)
    assert [s["clave"] for s in activos] == ["s1"]

    # 2. guardar_servicio autoincrementa servicio_id_num (max=2 -> 3)
    nuevo = repo.guardar_servicio("n1", {"clave": "s3", "nombre": "S3", "tarifa_unitaria": 3000})
    assert nuevo["servicio_id_num"] == 3

    # 3. obtener_o_crear_perfil actualiza nombre real cuando antes era "Cliente"
    perfil = repo.obtener_o_crear_perfil("n1", "573001112233", nombre="Laura Gomez")
    assert perfil["nombre"] == "Laura Gomez"

    # 4. registrar_agendamiento traduce 409 a ConflictoAgendaError
    with pytest.raises(ConflictoAgendaError):
        repo.registrar_agendamiento(
            "n1",
            {
                "id_cliente": "c1",
                "fecha_inicio": "2026-12-01T10:00:00-05:00",
                "fecha_fin": "2026-12-01T11:00:00-05:00",
            },
        )

    # 5. cancelar_agendamiento con ID no-UUID retorna False sin lanzar 500
    assert repo.cancelar_agendamiento("n1", "no-es-un-uuid") is False


def test_mile_cuentas_beauty_reglas_cotizacion_y_jornada_dividida() -> None:
    repo = RepositorioComercialSQLite(":memory:")
    cfg = _config_prueba()
    neg = repo.obtener_negocio("mile_cuentas_beauty")
    assert neg is not None
    assert neg["telefono_dueno"] == "573006900581"

    perfil = repo.obtener_o_crear_perfil("mile_cuentas_beauty", "573001112233", "Sofia Vargas")
    ctx = ContextoHerramientasNegocio(repo, neg, perfil, "WhatsApp", cfg)

    # 1. Catalogo exacto de 7 servicios
    cat = ctx.consultar_servicios()
    precios = {s["clave"]: s["precio"] for s in cat["servicios"]}
    assert precios == {
        "manicura_semipermanente": 35000.0,
        "kapping_gel": 45000.0,
        "dual_gel_premium": 60000.0,
        "soft_gel_premium": 70000.0,
        "pedi_spa_essential": 25000.0,
        "pedi_spa_care": 37000.0,
        "pedi_spa_luxury": 50000.0,
    }

    # 2. No permite combinar 2 servicios de manos en la misma cita
    err_combo = ctx.calcular_cotizacion(
        [
            {"servicio_clave": "manicura_semipermanente", "cantidad": 1},
            {"servicio_clave": "kapping_gel", "cantidad": 1},
        ]
    )
    assert "error" in err_combo

    # 3. Combinar 1 manos + 1 pies + diseno complejo + domicilio + tarjeta (+5%) + anticipo 20%
    cot = ctx.calcular_cotizacion(
        [
            {"servicio_clave": "manicura_semipermanente", "cantidad": 1},
            {"servicio_clave": "pedi_spa_care", "cantidad": 1},
        ],
        diseno_complejo=True,
        incluir_domicilio_base=True,
        pago_con_tarjeta=True,
    )
    # 35.000 + 37.000 + 25.000 + 15.000 = 112.000; +5% tarjeta = 5.600 -> 117.600; 20% anticipo = 23.520
    assert cot["subtotal"] == 112000.0
    assert cot["recargo_tarjeta_valor"] == 5600.0
    assert cot["total"] == 117600.0
    assert cot["anticipo_valor"] == 23520.0

    # 4. Reserva maxima de 1 dia de anticipacion: pasado manana rechazado
    pasado_manana = (date.today() + timedelta(days=2)).isoformat()
    disp_lejos = ctx.consultar_horarios_disponibles(pasado_manana, 60)
    assert disp_lejos["disponible"] is False

    # 5. Manana: jornada dividida (12:00-14:00 excluido), estado inicial pendiente y buffer de 15 min
    manana = (date.today() + timedelta(days=1)).isoformat()
    disp_manana = ctx.consultar_horarios_disponibles(manana, 60)
    assert disp_manana["dia_laboral"] is True
    assert not any(" 12:" in b or " 13:" in b for b in disp_manana["bloques_disponibles"])

    cita_almuerzo = ctx.agendar_cita(
        fecha_hora=f"{manana} 12:30",
        servicio_clave="manicura_semipermanente",
        nombre_cliente="Sofia Vargas",
        telefono_cliente="573001112233",
    )
    assert cita_almuerzo["confirmada"] is False
    assert "descanso" in cita_almuerzo["error"]

    cita_ok = ctx.agendar_cita(
        fecha_hora=f"{manana} 09:00",
        servicio_clave="manicura_semipermanente",
        nombre_cliente="Sofia Vargas",
        telefono_cliente="573001112233",
    )
    assert cita_ok["estado_cita"] == "pendiente"
    assert cita_ok["requiere_confirmacion_manual_pago"] is True
    assert cita_ok["anticipo_porcentaje"] == 20.0

    # 10:00 rechazada por buffer de 15 min; 10:15 aceptada
    cita_sin_buffer = ctx.agendar_cita(
        fecha_hora=f"{manana} 10:00",
        servicio_clave="pedi_spa_essential",
        nombre_cliente="Laura Ortiz",
        telefono_cliente="573009998877",
    )
    assert "error" in cita_sin_buffer

    cita_con_buffer = ctx.agendar_cita(
        fecha_hora=f"{manana} 10:15",
        servicio_clave="pedi_spa_essential",
        nombre_cliente="Laura Ortiz",
        telefono_cliente="573009998877",
    )
    assert cita_con_buffer["estado_cita"] == "pendiente"
