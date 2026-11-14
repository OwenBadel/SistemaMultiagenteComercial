"""
Suite de pruebas automatizadas del SistemaMultiagenteComercial:
1. Estructura organizacional MAS-ML (5 roles, control, 5 dependencias y reglas).
2. Datasets (>= 30 registros) y modelos Scikit-Learn (NLP, RandomForest, KNN).
3. Diagramas de secuencia FIPA-ACL (Escenario 1 alt/else, Escenario 2 alt/else,
   Agendamiento y Escalamiento Human-in-the-Loop).
4. Integracion real con Meta Platforms (WhatsApp Cloud API, Instagram Direct,
   Facebook Messenger), verificacion GET /webhook, POST /webhook y API humana.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

import httpx

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial
from src.integracion_meta.configuracion import ConfiguracionMeta
from src.integracion_meta.webhook_server import crear_app_meta
from src.ml.datasets import inicializar_datasets
from src.ml.modelos_ml import ModeloCotizacionesRF, ModeloNLPVentas, ModeloPromocionesKNN
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


def _extraer_performativas(mensajes_fipa: List[dict]) -> List[str]:
    """Extrae la secuencia ordenada de performativas FIPA-ACL registradas."""
    return [m["fipa_json"][":performative"] for m in mensajes_fipa]


def test_estructura_masml_roles_control_y_dependencias() -> None:
    """Verifica composicion de 5 roles, 4 controles, 5 dependencias y reglas MAS-ML."""
    df_t, df_o = inicializar_datasets(RAIZ / "data")
    entorno = EntornoVirtualAtencion(tarifas_historicas=df_t, historico_ofertas=df_o)
    org = SistemaMultiagenteComercial(entorno=entorno)

    assert set(org.roles.keys()) == {
        "Coordinador",
        "EjecutivoVentas",
        "LiquidadorCotizaciones",
        "AnalistaPromociones",
        "GestorAgenda",
    }
    for rol_sub in (
        "EjecutivoVentas",
        "AnalistaPromociones",
        "LiquidadorCotizaciones",
        "GestorAgenda",
    ):
        assert org.verificar_control("Coordinador", rol_sub) is True

    assert len(org.dependencias_formales) == 5
    assert org.verificar_dependencia("Coordinador", "LiquidadorCotizaciones")
    assert org.verificar_dependencia("Coordinador", "AnalistaPromociones")
    assert org.verificar_dependencia("EjecutivoVentas", "Coordinador")
    assert org.verificar_dependencia("GestorAgenda", "LiquidadorCotizaciones")
    assert org.verificar_dependencia("Coordinador", "GestorAgenda")

    assert org.aplicar_regla_escalamiento("EjecutivoVentas", "LiquidadorCotizaciones") is False
    assert org.aplicar_regla_escalamiento("EjecutivoVentas", "Coordinador") is True
    assert (
        org.aplicar_regla_canal_centralizado_promociones(
            "AnalistaPromociones", "EjecutivoVentas"
        )
        is False
    )
    assert (
        org.aplicar_regla_canal_centralizado_promociones(
            "AnalistaPromociones", "Coordinador"
        )
        is True
    )
    assert org.validar_prerequisito_agendamiento(False) is False
    assert org.validar_prerequisito_agendamiento(True) is True


def test_modelos_sklearn_y_datasets() -> None:
    """Valida datasets >=30 filas y metricas predictivas de los 3 modelos Scikit-Learn."""
    df_tarifas, df_ofertas = inicializar_datasets(RAIZ / "data")
    assert len(df_tarifas) >= 30
    assert len(df_ofertas) >= 30

    modelo_nlp = ModeloNLPVentas()
    modelo_rf = ModeloCotizacionesRF(df_tarifas)
    modelo_knn = ModeloPromocionesKNN(df_ofertas)

    assert modelo_rf.r2_entrenamiento > 0.85
    assert modelo_knn.r2_entrenamiento > 0.65

    pred_cot = modelo_rf.predecir_cotizacion(
        {
            "tipo_servicio": "ecommerce",
            "complejidad": 4,
            "horas_estimadas": 60,
            "urgencia_dias": 10,
            "incluye_mantenimiento": 1,
        }
    )
    assert pred_cot["valor_total"] > 0

    pred_promo = modelo_knn.predecir_descuento_optimo(
        monto_cotizado=pred_cot["valor_total"],
        nivel_indecision=0.72,
        complejidad=4,
        fidelidad_cliente=2,
    )
    assert 0.0 < pred_promo["descuento_porciento"] <= 20.0

    nlp_cot = modelo_nlp.analizar_mensaje(
        "Necesito cotizar desarrollo web de 45 horas con complejidad 3"
    )
    assert nlp_cot["intencion"] == "solicitud_cotizacion"
    assert nlp_cot["parametros_extraidos"]["horas_estimadas"] == 45
    assert nlp_cot["parametros_extraidos"]["complejidad"] == 3

    nlp_obj = modelo_nlp.analizar_mensaje("Me parece costoso para mi presupuesto")
    assert nlp_obj["intencion"] == "objecion_precio"
    assert nlp_obj["tono_cliente"] == "indeciso"

    nlp_hostil = modelo_nlp.analizar_mensaje("Esto es una estafa, quiero un humano ya")
    assert nlp_hostil["tono_cliente"] == "hostil"


def test_secuencia_1_cotizacion_multiturno_alt_y_else() -> None:
    """
    Demuestra Escenario 1 (FIPA-Request) completo en WhatsApp:
    1) Solicitud incompleta -> else [FAILURE parametros_incompletos -> REQUEST aclaracion]
    2) Respuesta con horas y complejidad -> alt [INFORM cotizacion_calculada -> INFORM entregar_cotizacion]
    """

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            ruta_data=RAIZ / "data",
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            res_incompleta = await servicio.procesar_mensaje_entrante_meta(
                canal="WhatsApp",
                id_cliente="573001112233",
                nombre_cliente="Laura Gomez",
                texto="Hola, necesito cotizar un desarrollo web",
            )
            perfs_1b = _extraer_performativas(res_incompleta["mensajes_fipa_turno"])
            assert "REQUEST" in perfs_1b
            assert "AGREE" in perfs_1b
            assert "FAILURE" in perfs_1b

            for msg_reg in res_incompleta["mensajes_fipa_turno"]:
                fipa = msg_reg["fipa_json"]
                assert fipa[":ontology"] == "Comercial-SMA"
                assert fipa[":protocol"] == "FIPA-Request"
                assert "<<ACL:" in msg_reg["etiqueta_diapositiva"]

            res_completa = await servicio.procesar_mensaje_entrante_meta(
                canal="WhatsApp",
                id_cliente="573001112233",
                nombre_cliente="Laura Gomez",
                texto="Son 45 horas estimadas y complejidad 3",
            )
            perfs_1a = _extraer_performativas(res_completa["mensajes_fipa_turno"])
            assert perfs_1a == ["REQUEST", "AGREE", "REQUEST", "AGREE", "INFORM", "INFORM"]
            assert res_completa["resultado"]["resultado"] == "cotizacion_entregada"
            assert res_completa["resultado"]["valor_total"] > 0
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_secuencia_2_negociacion_aceptada_y_rechazada() -> None:
    """
    Demuestra Escenario 2 (FIPA-Propose) en Instagram:
    - alt [acepta]: PROPOSE -> ACCEPT-PROPOSAL -> INFORM -> promocion presentada
    - else [rechaza]: PROPOSE -> REJECT-PROPOSAL -> REFUSE -> tarifa base mantenida
    """

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            ruta_data=RAIZ / "data",
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante_meta(
                canal="Instagram",
                id_cliente="ig_user_88",
                nombre_cliente="Carlos Ruiz",
                texto="Quiero cotizar un ecommerce de 60 horas con complejidad 4",
            )

            res_2a = await servicio.procesar_mensaje_entrante_meta(
                canal="Instagram",
                id_cliente="ig_user_88",
                nombre_cliente="Carlos Ruiz",
                texto="El precio esta algo costoso para mi presupuesto",
            )
            perfs_2a = _extraer_performativas(res_2a["mensajes_fipa_turno"])
            assert perfs_2a == [
                "REQUEST",
                "REQUEST",
                "PROPOSE",
                "ACCEPT-PROPOSAL",
                "INFORM",
                "INFORM",
            ]
            assert res_2a["resultado"]["resultado"] == "promocion_presentada"

            res_2b = await servicio.procesar_mensaje_entrante_meta(
                canal="Instagram",
                id_cliente="ig_user_88",
                nombre_cliente="Carlos Ruiz",
                texto="Necesito un 28% de descuento porque sigue demasiado caro",
            )
            perfs_2b = _extraer_performativas(res_2b["mensajes_fipa_turno"])
            assert perfs_2b == [
                "REQUEST",
                "REQUEST",
                "PROPOSE",
                "REJECT-PROPOSAL",
                "REFUSE",
            ]
            assert res_2b["resultado"]["resultado"] == "descuento_rechazado"
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_agendamiento_y_escalamiento_human_in_the_loop() -> None:
    """
    Valida Agendamiento FIPA-Query (bloque ocupado vs libre) y pausa
    Human-in-the-Loop cuando el cliente adopta tono hostil.
    """

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            ruta_data=RAIZ / "data",
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante_meta(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto="Cotizar consultoria cloud de 30 horas y complejidad 2",
            )

            res_ocupado = await servicio.procesar_mensaje_entrante_meta(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto="Quiero agendar una cita para el 2026-11-18 14:00",
            )
            assert "QUERY-IF" in _extraer_performativas(res_ocupado["mensajes_fipa_turno"])
            assert res_ocupado["resultado"]["resultado"] == "horarios_alternativos"

            res_libre = await servicio.procesar_mensaje_entrante_meta(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto="Entonces quiero reservar cita el 2026-11-18 10:00",
            )
            perfs_libre = _extraer_performativas(res_libre["mensajes_fipa_turno"])
            assert perfs_libre == [
                "REQUEST",
                "QUERY-IF",
                "INFORM",
                "REQUEST",
                "AGREE",
                "CONFIRM",
                "INFORM",
            ]
            assert res_libre["resultado"]["resultado"] == "cita_confirmada"
            assert (
                servicio.entorno.agendaVirtual.bloques["2026-11-18 10:00"]["estado"]
                == "OCUPADO"
            )

            res_hostil = await servicio.procesar_mensaje_entrante_meta(
                canal="WhatsApp",
                id_cliente="573009998877",
                nombre_cliente="Pedro Diaz",
                texto="Esto es una estafa, quiero reclamar con un humano ya",
            )
            assert res_hostil["alerta_humana"] is not None
            assert res_hostil["alerta_humana"]["estado"] == "alerta_recibida_por_profesional"
            perfil_pausado = servicio.entorno.perfilesClientes.perfiles["573009998877"]
            assert perfil_pausado["automatizacion_activa"] is False
            assert len(servicio.entorno.alertas_humanas) >= 1

            res_mientras_pausa = await servicio.procesar_mensaje_entrante_meta(
                canal="WhatsApp",
                id_cliente="573009998877",
                nombre_cliente="Pedro Diaz",
                texto="Sigo esperando respuesta",
            )
            assert res_mientras_pausa["resultado"]["resultado"] == "en_atencion_humana"

            envio_manual = await servicio.responder_como_humano(
                id_cliente="573009998877",
                texto="Hola Pedro, te atiende el supervisor comercial directamente.",
            )
            assert envio_manual["canal"] == "WhatsApp"

            perfil_reactivado = servicio.reactivar_automatizacion("573009998877")
            assert perfil_reactivado["automatizacion_activa"] is True
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_endpoints_webhook_meta_whatsapp_instagram_messenger() -> None:
    """
    Prueba los endpoints HTTP reales de FastAPI para Meta:
    - GET /webhook (verificacion con verify_token valido e invalido)
    - POST /webhook (payloads oficiales WhatsApp, Instagram Direct y Facebook Messenger)
    - Panel /, /api/estado, /api/humano/responder, /api/humano/reactivar y calendario
    """

    async def _run() -> None:
        config = ConfiguracionMeta(meta_verify_token="token_prueba_unicolombo")
        servicio = ServicioMultiagenteComercial(
            ruta_data=RAIZ / "data",
            config_meta=config,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        app = crear_app_meta(servicio)

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                r_ok = await client.get(
                    "/webhook",
                    params={
                        "hub.mode": "subscribe",
                        "hub.verify_token": "token_prueba_unicolombo",
                        "hub.challenge": "99887766",
                    },
                )
                assert r_ok.status_code == 200
                assert r_ok.text == "99887766"

                r_bad = await client.get(
                    "/webhook",
                    params={
                        "hub.mode": "subscribe",
                        "hub.verify_token": "token_erroneo",
                        "hub.challenge": "123",
                    },
                )
                assert r_bad.status_code == 403

                r_panel = await client.get("/")
                assert r_panel.status_code == 200
                assert "SistemaMultiagenteComercial" in r_panel.text

                payload_wa = {
                    "object": "whatsapp_business_account",
                    "entry": [
                        {
                            "changes": [
                                {
                                    "value": {
                                        "contacts": [
                                            {
                                                "wa_id": "573101234567",
                                                "profile": {"name": "Ana Sofia"},
                                            }
                                        ],
                                        "messages": [
                                            {
                                                "from": "573101234567",
                                                "type": "text",
                                                "text": {
                                                    "body": "Hola, quiero cotizar una app movil de 50 horas complejidad 3"
                                                },
                                            }
                                        ],
                                    }
                                }
                            ]
                        }
                    ],
                }
                r_wa = await client.post("/webhook", json=payload_wa)
                assert r_wa.status_code == 200
                data_wa = r_wa.json()
                assert data_wa["mensajes_procesados"] == 1
                assert data_wa["resultados"][0]["canal"] == "WhatsApp"
                assert data_wa["resultados"][0]["resultado"]["resultado"] == "cotizacion_entregada"

                payload_ig = {
                    "object": "instagram",
                    "entry": [
                        {
                            "messaging": [
                                {
                                    "sender": {"id": "ig_9001"},
                                    "message": {"text": "Hola, que servicios ofrecen?"},
                                }
                            ]
                        }
                    ],
                }
                r_ig = await client.post("/webhook", json=payload_ig)
                assert r_ig.status_code == 200
                data_ig = r_ig.json()
                assert data_ig["mensajes_procesados"] == 1
                assert data_ig["resultados"][0]["canal"] == "Instagram"
                assert data_ig["resultados"][0]["resultado"]["accion"] == "catalogo_enviado"

                payload_fb = {
                    "object": "page",
                    "entry": [
                        {
                            "messaging": [
                                {
                                    "sender": {"id": "fb_7001"},
                                    "message": {"text": "Esto es una estafa, pasame a un humano ya"},
                                }
                            ]
                        }
                    ],
                }
                r_fb = await client.post("/webhook", json=payload_fb)
                assert r_fb.status_code == 200
                assert r_fb.json()["resultados"][0]["alerta_humana"] is not None

                r_resp = await client.post(
                    "/api/humano/responder",
                    json={"id_cliente": "fb_7001", "texto": "Atendido por asesor humano."},
                )
                assert r_resp.status_code == 200
                assert r_resp.json()["canal"] == "Facebook"

                r_react = await client.post(
                    "/api/humano/reactivar",
                    json={"id_cliente": "fb_7001"},
                )
                assert r_react.status_code == 200
                assert r_react.json()["perfil"]["automatizacion_activa"] is True

                r_cal = await client.post(
                    "/api/calendario/disponibilidad",
                    json={"fecha_hora": "2026-11-19 09:00", "estado": "OCUPADO", "cliente": "VIP"},
                )
                assert r_cal.status_code == 200

                r_estado = await client.get("/api/estado")
                assert r_estado.status_code == 200
                estado = r_estado.json()
                assert len(estado["agentes"]) == 5
                assert len(estado["mensajes_fipa"]) > 0
        finally:
            await servicio.detener()

    asyncio.run(_run())
