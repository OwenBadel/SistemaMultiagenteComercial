"""
Suite de pruebas automatizadas del SistemaMultiagenteComercial:
1. Estructura organizacional MAS-ML (5 roles, control, 5 dependencias y reglas).
2. Datasets (>= 30 registros) y modelos Scikit-Learn (NLP, RandomForest, KNN).
3. Diagramas de secuencia FIPA-ACL (Escenario 1 alt/else, Escenario 2 alt/else,
   Agendamiento y Escalamiento Human-in-the-Loop).
4. Integracion API oficial para Botpress (WhatsApp, Webchat, Messenger, Instagram),
   seguridad opcional por API Key y consola Human-in-the-Loop.

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
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.integracion_botpress.servidor_botpress import crear_app_botpress
from src.ml.datasets import generar_historico_ofertas, generar_tarifas_historicas
from src.ml.modelos_ml import ModeloCotizacionesRF, ModeloNLPVentas, ModeloPromocionesKNN
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.persistencia.repositorio import RepositorioSQLite, RepositorioSupabase


def _extraer_performativas(mensajes_fipa: List[dict]) -> List[str]:
    """Extrae la secuencia ordenada de performativas FIPA-ACL registradas."""
    return [m["fipa_json"][":performative"] for m in mensajes_fipa]


def test_estructura_masml_roles_control_y_dependencias() -> None:
    """Verifica composicion de 5 roles, 4 controles, 5 dependencias y reglas MAS-ML."""
    df_t = generar_tarifas_historicas()
    df_o = generar_historico_ofertas()
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
    df_tarifas = generar_tarifas_historicas()
    df_ofertas = generar_historico_ofertas()
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
    Demuestra Escenario 1 (FIPA-Request) completo desde Botpress:
    1) Solicitud incompleta -> else [FAILURE parametros_incompletos -> REQUEST aclaracion]
    2) Respuesta con horas y complejidad -> alt [INFORM cotizacion_calculada -> INFORM entregar_cotizacion]
    """

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            res_incompleta = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573001112233",
                nombre_cliente="Laura Gomez",
                texto="Hola, necesito cotizar un desarrollo web",
            )
            perfs_1b = _extraer_performativas(res_incompleta["mensajes_fipa_turno"])
            assert "REQUEST" in perfs_1b
            assert "AGREE" in perfs_1b
            assert "FAILURE" in perfs_1b
            assert len(res_incompleta["respuesta"]) > 10

            for msg_reg in res_incompleta["mensajes_fipa_turno"]:
                fipa = msg_reg["fipa_json"]
                assert fipa[":ontology"] == "Comercial-SMA"
                assert fipa[":protocol"] == "FIPA-Request"
                assert "<<ACL:" in msg_reg["etiqueta_diapositiva"]

            res_completa = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573001112233",
                nombre_cliente="Laura Gomez",
                texto="Son 45 horas estimadas y complejidad 3",
            )
            perfs_1a = _extraer_performativas(res_completa["mensajes_fipa_turno"])
            assert perfs_1a == ["REQUEST", "AGREE", "REQUEST", "AGREE", "INFORM", "INFORM"]
            assert res_completa["resultado"]["resultado"] == "cotizacion_entregada"
            assert res_completa["resultado"]["valor_total"] > 0
            assert "$" in res_completa["respuesta"]
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_secuencia_2_negociacion_aceptada_y_rechazada() -> None:
    """
    Demuestra Escenario 2 (FIPA-Propose) en canal Webchat:
    - alt [acepta]: PROPOSE -> ACCEPT-PROPOSAL -> INFORM -> promocion presentada
    - else [rechaza]: PROPOSE -> REJECT-PROPOSAL -> REFUSE -> tarifa base mantenida
    """

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="Webchat",
                id_cliente="web_user_88",
                nombre_cliente="Carlos Ruiz",
                texto="Quiero cotizar un ecommerce de 60 horas con complejidad 4",
            )

            res_2a = await servicio.procesar_mensaje_entrante(
                canal="Webchat",
                id_cliente="web_user_88",
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

            res_2b = await servicio.procesar_mensaje_entrante(
                canal="Webchat",
                id_cliente="web_user_88",
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
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto="Cotizar consultoria cloud de 30 horas y complejidad 2",
            )

            res_ocupado = await servicio.procesar_mensaje_entrante(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto="Quiero agendar una cita para el 2026-11-18 14:00",
            )
            assert "QUERY-IF" in _extraer_performativas(res_ocupado["mensajes_fipa_turno"])
            assert res_ocupado["resultado"]["resultado"] == "horarios_alternativos"

            res_libre = await servicio.procesar_mensaje_entrante(
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

            res_hostil = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573009998877",
                nombre_cliente="Pedro Diaz",
                texto="Esto es una estafa, quiero reclamar con un humano ya",
            )
            assert res_hostil["pausado_humano"] is True
            assert res_hostil["notificar_dueno"] is True
            assert res_hostil["responder_al_cliente"] is True
            assert "573009998877" in res_hostil["mensaje_notificacion_dueno"]
            assert "Pedro Diaz" in res_hostil["mensaje_notificacion_dueno"]
            assert res_hostil["alerta_humana"] is not None
            assert res_hostil["alerta_humana"]["estado"] == "alerta_recibida_por_profesional"
            perfil_pausado = servicio.entorno.perfilesClientes.perfiles["573009998877"]
            assert perfil_pausado["automatizacion_activa"] is False
            assert len(servicio.entorno.alertas_humanas) >= 1

            # Segundo mensaje en el mismo chat pausado: bot en silencio, sin re-alertar al dueno
            res_mientras_pausa = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573009998877",
                nombre_cliente="Pedro Diaz",
                texto="Sigo esperando respuesta, esto es un robo",
            )
            assert res_mientras_pausa["resultado"]["resultado"] == "en_atencion_humana"
            assert res_mientras_pausa["pausado_humano"] is True
            assert res_mientras_pausa["responder_al_cliente"] is False
            assert res_mientras_pausa["notificar_dueno"] is False
            assert res_mientras_pausa["respuesta"] == ""

            # Otro cliente distinto sigue siendo atendido normalmente (pausa aislada por chat)
            res_otro_cliente = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573008887766",
                nombre_cliente="Maria Lopez",
                texto="Hola, quiero cotizar desarrollo web de 35 horas complejidad 2",
            )
            assert res_otro_cliente["pausado_humano"] is False
            assert res_otro_cliente["responder_al_cliente"] is True
            assert res_otro_cliente["accion"] == "cotizacion_entregada"

            perfil_reactivado = servicio.reactivar_automatizacion("573009998877")
            assert perfil_reactivado["automatizacion_activa"] is True
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_endpoints_botpress_y_panel_humano() -> None:
    """
    Prueba los endpoints HTTP reales de FastAPI para Botpress:
    - GET /, GET /health
    - POST /api/botpress/mensaje (formato en espanol y aliases nativos Botpress)
    - Proteccion opcional con BOTPRESS_API_KEY
    - Panel /api/estado, /api/humano/responder, /api/humano/reactivar y calendario
    """

    async def _run() -> None:
        config = ConfiguracionBotpress(
            botpress_api_key="clave_botpress_2026",
            owner_whatsapp_number="573005551122",
        )
        servicio = ServicioMultiagenteComercial(
            config_botpress=config,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        app = crear_app_botpress(servicio)

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                r_panel = await client.get("/")
                assert r_panel.status_code == 200
                assert "SistemaMultiagenteComercial" in r_panel.text

                r_health = await client.get("/health")
                assert r_health.status_code == 200
                assert r_health.json()["agentes_activos"] == 5

                r_unauth = await client.post(
                    "/api/botpress/mensaje",
                    json={"canal": "WhatsApp", "id_cliente": "u1", "texto": "Hola"},
                )
                assert r_unauth.status_code == 401

                headers = {"x-api-key": "clave_botpress_2026"}

                r_wa = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "canal": "WhatsApp",
                        "id_cliente": "573101234567",
                        "nombre": "Ana Sofia",
                        "texto": "Hola, quiero cotizar una app movil de 50 horas complejidad 3",
                    },
                )
                assert r_wa.status_code == 200
                data_wa = r_wa.json()
                assert data_wa["canal"] == "WhatsApp"
                assert data_wa["accion"] == "cotizacion_entregada"
                assert data_wa["pausado_humano"] is False
                assert "$" in data_wa["respuesta"]

                r_web = await client.post(
                    "/api/botpress/mensaje",
                    headers={"Authorization": "Bearer clave_botpress_2026"},
                    json={
                        "channel": "Webchat",
                        "userId": "bp_web_9001",
                        "text": "Hola, que servicios ofrecen?",
                    },
                )
                assert r_web.status_code == 200
                data_web = r_web.json()
                assert data_web["canal"] == "Webchat"
                assert data_web["accion"] == "catalogo_enviado"
                assert len(data_web["respuesta"]) > 20

                r_fb = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "canal": "Facebook",
                        "id_cliente": "fb_7001",
                        "texto": "Esto es una estafa, pasame a un humano ya",
                    },
                )
                assert r_fb.status_code == 200
                data_fb = r_fb.json()
                assert data_fb["pausado_humano"] is True
                assert data_fb["notificar_dueno"] is True
                assert data_fb["telefono_dueno"] == "573005551122"
                assert "fb_7001" in data_fb["mensaje_notificacion_dueno"]
                assert data_fb["alerta_humana"] is not None

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


def test_negocio_distinto_dinamico_desde_bd() -> None:
    """
    Demuestra que los agentes no tienen logica de negocio quemada: al conectar
    un negocio completamente distinto en BD ('estudio_fotografia' cotizando por
    sesiones y sin requerir complejidad en retrato), los mismos 5 agentes cotizan
    y envian catalogo de fotografia sin cambiar codigo.
    """
    repo = RepositorioSQLite(
        ruta_db=":memory:",
        negocio_id="estudio_fotografia",
        auto_seed=False,
    )
    conn = repo._conn
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO negocios (
            id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
            etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
            descuento_maximo, mensaje_catalogo_intro
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "estudio_fotografia",
            "Estudio Visual Independiente",
            "Fotografia Comercial",
            "COP",
            "sesiones",
            "sesiones_requeridas",
            "complejidad",
            "573007776655",
            0.40,
            0.25,
            12.0,
            "Aqui tienes nuestros paquetes de fotografia",
        ),
    )
    cur.execute(
        """
        INSERT INTO servicios (
            negocio_id, clave, servicio_id_num, nombre, descripcion,
            tarifa_unitaria, palabras_clave, requiere_complejidad,
            costo_adicional_mantenimiento
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "estudio_fotografia",
            "retrato_corporativo",
            1,
            "Retrato Corporativo y Marca Personal",
            "Sesion profesional con entrega digital",
            320_000.0,
            '["retrato", "corporativo", "foto"]',
            0,
            120_000.0,
        ),
    )
    cur.execute(
        """
        INSERT INTO promociones_vigentes (negocio_id, codigo, descripcion, descuento_max, activa)
        VALUES ('estudio_fotografia', 'PACK_10', 'Descuento multi-sesion', 10.0, 1)
        """
    )
    for i in range(1, 16):
        cur.execute(
            """
            INSERT INTO tarifas_historicas (
                negocio_id, id_cotizacion, tipo_servicio, tipo_servicio_id,
                complejidad, cantidad_unidades, urgencia_dias, incluye_adicional,
                costo_base, costo_total, margen_estimado
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "estudio_fotografia",
                f"FOT-{i:03d}",
                "retrato_corporativo",
                1,
                3,
                1 + (i % 6),
                10,
                i % 2,
                float((1 + (i % 6)) * 320_000),
                float((1 + (i % 6)) * 320_000 + (i % 2) * 120_000),
                0.40,
            ),
        )
    for i in range(1, 16):
        cur.execute(
            """
            INSERT INTO historico_ofertas (
                negocio_id, id_oferta, monto_cotizado, nivel_indecision,
                complejidad, fidelidad_cliente, porcentaje_descuento_optimo,
                cerrado_exitosamente
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("estudio_fotografia", f"OFE-FOT-{i:03d}", 600_000 + i * 50_000, 0.70, 3, 1, 8.0, 1),
        )
    cur.execute(
        "INSERT INTO agenda_bloques VALUES ('estudio_fotografia', '2026-11-20 09:00', 'LIBRE', NULL)"
    )
    conn.commit()

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            res_cat = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_1",
                nombre_cliente="Camila",
                texto="Hola, que servicios ofrecen?",
            )
            assert "Retrato Corporativo" in res_cat["respuesta"]
            assert "sesiones" in res_cat["respuesta"]

            res_cot = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_1",
                nombre_cliente="Camila",
                texto="Quiero cotizar 3 sesiones de retrato corporativo",
            )
            assert res_cot["accion"] == "cotizacion_entregada"
            assert "Retrato Corporativo" in res_cot["respuesta"]
            assert res_cot["resultado"]["valor_total"] > 0
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_repositorio_supabase_rest_api() -> None:
    """Verifica el cliente REST HTTPS de Supabase sin depender de red externa."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path.endswith("/negocios"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "negocio_nube",
                        "nombre_negocio": "Consultora Nube",
                        "rubro": "Asesoria",
                        "moneda": "USD",
                        "unidad_cantidad": "horas",
                        "etiqueta_cantidad": "horas_estimadas",
                        "etiqueta_complejidad": "complejidad",
                        "telefono_dueno": "573001119988",
                        "margen_base": 0.35,
                        "margen_minimo": 0.22,
                        "descuento_maximo": 15.0,
                        "mensaje_catalogo_intro": "Catalogo nube",
                    }
                ],
            )
        if request.method == "GET" and path.endswith("/servicios"):
            return httpx.Response(
                200,
                json=[
                    {
                        "clave": "auditoria",
                        "servicio_id_num": 1,
                        "nombre": "Auditoria Express",
                        "descripcion": "Revision integral",
                        "tarifa_unitaria": 120.0,
                        "palabras_clave": ["auditoria"],
                        "requiere_complejidad": True,
                        "costo_adicional_mantenimiento": 50.0,
                    }
                ],
            )
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    repo = RepositorioSupabase(
        supabase_url="https://demo-proyecto.supabase.co/rest/v1",
        supabase_key="service_role_demo",
        negocio_id="negocio_nube",
        http_client=client,
    )
    assert repo.base_rest == "https://demo-proyecto.supabase.co/rest/v1"
    cfg = repo.obtener_configuracion_negocio()
    assert cfg["moneda"] == "USD"
    assert cfg["telefono_dueno"] == "573001119988"

    cat = repo.obtener_catalogo_servicios()
    assert "auditoria" in cat
    assert cat["auditoria"]["tarifa_unitaria"] == 120.0

    repo.guardar_bloque_agenda("2026-11-22 10:00", "OCUPADO", "Cliente Nube")
    repo.registrar_alerta_humana(
        {
            "timestamp": "2026-11-18T12:00:00",
            "id_cliente": "u1",
            "nombre": "Test",
            "canal": "WhatsApp",
            "motivo_critico": "reclamo",
            "historial_charla": [],
            "estado": "alerta_recibida_por_profesional",
            "resuelta": False,
        }
    )
    repo.resolver_alertas_cliente("u1")

    try:
        RepositorioSupabase(
            supabase_url="sb_publishable_demo_key",
            supabase_key="sb_secret_demo",
        )
        assert False, "Debio rechazar una API key en SUPABASE_URL"
    except ValueError as err:
        assert "Project Settings -> Data API" in str(err)

