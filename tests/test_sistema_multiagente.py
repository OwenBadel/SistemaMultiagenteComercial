"""
Suite de pruebas automatizadas del SistemaMultiagenteComercial:
1. Estructura organizacional MAS-ML (5 roles, control, 5 dependencias y reglas).
2. Datasets (>= 30 registros) y modelos Scikit-Learn (NLP, RandomForest, KNN).
3. Diagramas de secuencia FIPA-ACL (Escenario 1 alt/else, Escenario 2 alt/else,
   Agendamiento y Escalamiento Human-in-the-Loop).
4. Integracion API oficial para Botpress (WhatsApp, Webchat, Messenger, Instagram),
   seguridad opcional por API Key y consola Human-in-the-Loop.
"""
from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import List
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import httpx
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.entorno.entorno_virtual import EntornoVirtualAtencion, Informacion
from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.integracion_botpress.servidor_botpress import (
    PeticionMensajeBotpress,
    crear_app_botpress,
)
from src.ml.datasets import generar_historico_ofertas, generar_tarifas_historicas
from src.ml.generador_llm import GeneradorDialogoGemini
from src.ml.modelos_ml import (
    ModeloCotizacionesRF,
    ModeloNLPVentas,
    ModeloPromocionesKNN,
    _extraer_fecha_hora,
)
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.persistencia.google_calendar import ClienteGoogleCalendar, ErrorGoogleCalendar
from src.persistencia.repositorio import (
    ConflictoAgendamientoError,
    RepositorioSQLite,
    RepositorioSupabase,
)
from src.reglas_comerciales import promocion_elegible


def _extraer_performativas(mensajes_fipa: List[dict]) -> List[str]:
    """Extrae la secuencia ordenada de performativas FIPA-ACL registradas."""
    return [m["fipa_json"][":performative"] for m in mensajes_fipa]


async def _capturar_datos_cliente_reserva(
    servicio: ServicioMultiagenteComercial,
    canal: str,
    id_cliente: str,
    conversation_id: str | None = None,
    nombre: str = "Andrea Prueba",
) -> dict:
    resultado: dict = {}
    for texto in (nombre, "1020304050", "+573001112233"):
        resultado = await servicio.procesar_mensaje_entrante(
            canal=canal,
            id_cliente=id_cliente,
            nombre_cliente=nombre,
            texto=texto,
            conversation_id=conversation_id,
        )
    return resultado


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

    nlp_coloquial = modelo_nlp.analizar_mensaje("oye compadre que vendes")
    assert nlp_coloquial["intencion"] == "consulta_servicio"
    assert nlp_coloquial["tono_cliente"] == "interesado"

    nlp_rechazo = modelo_nlp.analizar_mensaje("No voy a comprar, gracias, excede mi presupuesto")
    assert nlp_rechazo["intencion"] == "rechazo_compra"
    assert nlp_rechazo["tono_cliente"] == "indeciso"


def test_extrae_cantidades_y_complejidad_en_espanol_segun_unidad() -> None:
    catalogo = {
        "sesion_producto": {
            "id": 1,
            "nombre": "Fotografia de Producto",
            "palabras_clave": ["producto"],
            "requiere_complejidad": True,
        }
    }
    modelo_sesiones = ModeloNLPVentas(
        catalogo_servicios=catalogo,
        unidad_cantidad="sesiones",
        etiqueta_cantidad="sesiones_requeridas",
    )
    for texto, esperado in (
        ("Una sesión", 1),
        ("Una sola sesión", 1),
        ("1 sesión", 1),
        ("1 sola sesión", 1),
        ("Necesito dos sesiones", 2),
        ("Dos sesiones", 2),
        ("2 sesiones", 2),
        ("Tres sesiones", 3),
        ("3 sesiones", 3),
    ):
        parametros = modelo_sesiones._extraer_parametros_texto(texto)
        assert parametros["cantidad_unidades"] == esperado, texto
        assert parametros["sesiones_requeridas"] == esperado
        assert "horas_estimadas" not in parametros

    modelo_horas = ModeloNLPVentas(unidad_cantidad="horas")
    assert modelo_horas._extraer_parametros_texto("Una hora")["horas_estimadas"] == 1
    assert modelo_horas._extraer_parametros_texto("2 horas")["horas_estimadas"] == 2
    modelo_unidades = ModeloNLPVentas(unidad_cantidad="unidades")
    assert (
        modelo_unidades._extraer_parametros_texto("5 unidades")["cantidad_unidades"]
        == 5
    )
    modelo_fotografias = ModeloNLPVentas(unidad_cantidad="fotografías")
    assert (
        modelo_fotografias._extraer_parametros_texto("10 fotografías")[
            "cantidad_unidades"
        ]
        == 10
    )
    assert not modelo_sesiones._extraer_parametros_texto("Necesito una sola")
    assert not modelo_sesiones._extraer_parametros_texto("Necesito cinco horas")

    for texto, esperado in (
        ("Complejidad 1", 1),
        ("Nivel de complejidad 1", 1),
        ("La complejidad es 1", 1),
        ("La complejidad es uno", 1),
        ("Complejidad de nivel 3", 3),
        ("Nivel tres", 3),
    ):
        assert (
            modelo_sesiones._extraer_parametros_texto(texto)["complejidad"]
            == esperado
        )
    assert "complejidad" not in modelo_sesiones._extraer_parametros_texto(
        "Complejidad 6"
    )
    assert "complejidad" not in modelo_sesiones._extraer_parametros_texto(
        "Complejidad media"
    )
    modelo_equivalencias = ModeloNLPVentas(
        unidad_cantidad="sesiones",
        equivalencias_complejidad={"media": 3},
    )
    assert (
        modelo_equivalencias._extraer_parametros_texto("Complejidad media")[
            "complejidad"
        ]
        == 3
    )


def test_contexto_de_cotizacion_interpreta_respuesta_corta_y_productos_separados() -> None:
    catalogo = {
        "sesion_producto": {
            "id": 1,
            "nombre": "Fotografia de Producto E-Commerce",
            "palabras_clave": ["producto", "e-commerce"],
            "unidad_cantidad": "sesiones",
            "etiqueta_cantidad": "sesiones_requeridas",
            "requiere_complejidad": True,
        }
    }
    modelo = ModeloNLPVentas(
        catalogo_servicios=catalogo,
        unidad_cantidad="sesiones",
        etiqueta_cantidad="sesiones_requeridas",
    )
    parametros = modelo._extraer_parametros_texto(
        "Necesito una sola sesión y son 10 productos"
    )
    assert parametros["sesiones_requeridas"] == 1
    assert parametros["cantidad_productos"] == 10

    analisis = modelo.analizar_mensaje(
        "1",
        contexto_tipo="reclamo_critico",
        campos_pendientes=["complejidad"],
        parametros_acumulados={
            "tipo_servicio": "sesion_producto",
            "sesiones_requeridas": 1,
        },
    )
    assert analisis["intencion"] == "solicitud_cotizacion"
    assert analisis["parametros_extraidos"]["complejidad"] == 1
    assert not analisis["escalamiento_humano"]
    assert analisis["probabilidad_modelo"] == analisis["confianza"]
    assert analisis["probabilidad_modelo"] < 0.78

    reclamo = modelo.analizar_mensaje("Esto es una estafa, quiero un humano")
    assert reclamo["escalamiento_humano"]
    assert reclamo["intencion"] == "reclamo_critico"
    assert not modelo.analizar_mensaje("Hola, ¿qué servicios ofrecen?")[
        "escalamiento_humano"
    ]


def test_identidad_botpress_no_inventa_fallback_y_separa_clientes() -> None:
    turno_1 = PeticionMensajeBotpress(
        channel="Webchat",
        userId="usuario-estable",
        conversationId="conversacion-turno-1",
        text="Primero",
    ).normalizar()
    turno_2 = PeticionMensajeBotpress(
        channel="Webchat",
        userId="usuario-estable",
        conversationId="conversacion-turno-2",
        text="Segundo",
    ).normalizar()
    otro_cliente = PeticionMensajeBotpress(
        channel="Webchat",
        userId="otro-usuario",
        text="Otro",
    ).normalizar()
    solo_conversacion = PeticionMensajeBotpress(
        channel="Webchat",
        conversationId="conversacion-persistente",
        text="Turno",
    ).normalizar()
    usuario_generico_con_conversacion = PeticionMensajeBotpress(
        channel="Webchat",
        userId="Anonymous User (User)",
        conversationId="conversacion-prueba",
        text="Turno",
    ).normalizar()
    sin_identidad = PeticionMensajeBotpress(text="Sin ID").normalizar()
    solo_id_evento = PeticionMensajeBotpress(
        **{"id": "evento-por-mensaje", "text": "No usar id de evento"}
    ).normalizar()
    id_generico = PeticionMensajeBotpress(
        channel="Webchat",
        userId="Anonymous User (User)",
        text="Sin ID estable ni conversación",
    ).normalizar()
    mismo_usuario_otro_canal = PeticionMensajeBotpress(
        channel="WhatsApp",
        userId="usuario-estable",
        text="Mismo valor en otro canal",
    ).normalizar()

    assert turno_1["id_cliente"] == turno_2["id_cliente"]
    assert turno_1["id_cliente"] == "botpress:webchat:user:usuario-estable"
    assert turno_1["fuente_identidad"] == "userId"
    assert turno_1["alcance_identidad"] == "cliente"
    assert turno_1["id_cliente"] != otro_cliente["id_cliente"]
    assert solo_conversacion["id_cliente"] == (
        "botpress:webchat:conversation:conversacion-persistente"
    )
    assert solo_conversacion["alcance_identidad"] == "conversacion"
    assert usuario_generico_con_conversacion["id_cliente"] == (
        "botpress:webchat:conversation:conversacion-prueba"
    )
    assert sin_identidad["id_cliente"] == ""
    assert solo_id_evento["id_cliente"] == ""
    assert id_generico["id_cliente"] == ""
    assert turno_1["id_cliente"] != mismo_usuario_otro_canal["id_cliente"]
    id_explicito = PeticionMensajeBotpress(
        channel="Webchat", id_cliente="cliente-canonico", userId="ignored-user"
    ).normalizar()
    assert id_explicito["id_cliente"] == "cliente-canonico"


def test_preguntas_de_catalogo_no_se_clasifican_como_cotizacion() -> None:
    repo = RepositorioSQLite(":memory:")
    modelo_general = ModeloNLPVentas(
        catalogo_servicios=repo.obtener_catalogo_servicios()
    )
    for texto in (
        "Hola, ¿qué servicios ofrecen?",
        "Quiero conocer sus servicios",
        "¿Qué tipos de servicios tienen?",
    ):
        analisis = modelo_general.analizar_mensaje(texto)
        assert analisis["intencion"] == "consulta_servicio", texto
        assert "tipo_servicio" not in analisis["parametros_extraidos"], texto

    catalogo_fotografia = {
        "sesion_producto": {
            "id": 1,
            "nombre": "Fotografia de Producto E-Commerce",
            "palabras_clave": ["producto", "e-commerce"],
            "requiere_complejidad": True,
        }
    }
    modelo_fotografia = ModeloNLPVentas(
        catalogo_servicios=catalogo_fotografia,
        unidad_cantidad="sesiones",
        etiqueta_cantidad="sesiones_requeridas",
    )
    for texto in (
        "Me interesa fotografía de producto",
        "Quiero cotizar fotografía de producto",
    ):
        analisis = modelo_fotografia.analizar_mensaje(texto)
        assert analisis["intencion"] == "solicitud_cotizacion", texto
        assert analisis["parametros_extraidos"]["tipo_servicio"] == "sesion_producto"


def test_extrae_fechas_relativas_en_hora_local_de_negocio() -> None:
    ahora = datetime(2026, 11, 23, 19, 16, tzinfo=ZoneInfo("America/Bogota"))

    def fecha(texto: str) -> str:
        resultado = _extraer_fecha_hora(texto, ahora)
        assert resultado is not None
        return resultado.strftime("%Y-%m-%d %H:%M")

    assert fecha("quiero agendar mañana") == "2026-11-24 00:00"
    assert fecha("agenda ayer") == "2026-11-22 00:00"
    assert fecha("reserva en dos días a las 3 pm") == "2026-11-25 15:00"
    assert fecha("reserva dentro de 2 días") == "2026-11-25 00:00"


def test_extrae_fechas_naturales_iso_y_formatos_espanoles() -> None:
    ahora = datetime(2026, 11, 23, 19, 16, tzinfo=ZoneInfo("America/Bogota"))
    casos = {
        "25 de noviembre a las 9am": "2026-11-25 09:00",
        "25 de noviembre del 2026 a las 9:00 a.m.": "2026-11-25 09:00",
        "25 noviembre 2026 09:00": "2026-11-25 09:00",
        "25/11/2026 9:00": "2026-11-25 09:00",
        "25-11-2026 9:00": "2026-11-25 09:00",
        "Mañana a las 9am": "2026-11-24 09:00",
        "Próximo viernes a las 3pm": "2026-11-27 15:00",
        "2026-11-25 09:00": "2026-11-25 09:00",
    }

    for texto, esperado in casos.items():
        fecha = _extraer_fecha_hora(texto, ahora)
        assert fecha is not None, texto
        assert fecha.strftime("%Y-%m-%d %H:%M") == esperado
        assert str(fecha.tzinfo) == "America/Bogota"

    siguiente_anio = _extraer_fecha_hora(
        "25 de noviembre a las 9am",
        datetime(2026, 11, 26, 19, 16, tzinfo=ZoneInfo("America/Bogota")),
    )
    assert siguiente_anio is not None
    assert siguiente_anio.strftime("%Y-%m-%d %H:%M") == "2026-11-25 09:00"


def test_fechas_horas_y_anios_explicitos_en_bogota() -> None:
    ahora = datetime(2026, 12, 31, 16, 0, tzinfo=ZoneInfo("America/Bogota"))
    casos = {
        "mañana a las 10 de la mañana": "2027-01-01 10:00",
        "mañana 8 a. m.": "2027-01-01 08:00",
        "mañana 8:00am": "2027-01-01 08:00",
        "mañana a las 20:00": "2027-01-01 20:00",
        "15 de febrero del año que viene a las 9": "2027-02-15 09:00",
        "15 de febrero de 2028 a las 9:00": "2028-02-15 09:00",
    }
    for texto, esperado in casos.items():
        fecha = _extraer_fecha_hora(texto, ahora)
        assert fecha is not None, texto
        assert fecha.strftime("%Y-%m-%d %H:%M") == esperado
        assert str(fecha.tzinfo) == "America/Bogota"
    assert _extraer_fecha_hora("8:00am", ahora) is None


def test_respuesta_corta_de_horario_no_es_objecion_de_precio() -> None:
    analisis = ModeloNLPVentas().analizar_mensaje(
        "a las 9 está bien",
        contexto_tipo="solicitud_agendamiento",
        parametros_acumulados={"fecha_agendamiento_fecha": "2027-01-02"},
    )
    assert analisis["intencion"] == "solicitud_agendamiento"
    assert analisis["intencion"] != "objecion_precio"
    assert analisis["parametros_extraidos"]["fecha_hora"] == "2027-01-02 09:00"
    assert "descuento_solicitado_cliente" not in analisis["parametros_extraidos"]

    cambio_fecha = ModeloNLPVentas().analizar_mensaje(
        "el 3 de enero",
        contexto_tipo="solicitud_agendamiento",
        parametros_acumulados={
            "fecha_agendamiento_pendiente": "2027-01-02 09:00"
        },
    )
    assert cambio_fecha["parametros_extraidos"]["fecha_hora"] == "2027-01-03 09:00"


def test_identificadores_y_respuestas_numericas_no_son_objeciones_de_precio() -> None:
    modelo = ModeloNLPVentas()
    for texto in ("1143414217", "573001112233", "280000", "30", "2"):
        analisis = modelo.analizar_mensaje(texto)
        assert analisis["intencion"] != "objecion_precio", texto

    respuesta_complejidad = modelo.analizar_mensaje(
        "2", campos_pendientes=["complejidad"]
    )
    assert respuesta_complejidad["parametros_extraidos"].get("complejidad") == 2


def test_fecha_sin_hora_se_conserva_como_parcial() -> None:
    analisis = ModeloNLPVentas().analizar_mensaje(
        "Quiero agendar cita para mañana",
        contexto_tipo="solicitud_agendamiento",
    )
    assert "fecha_agendamiento_fecha" in analisis["parametros_extraidos"]
    assert "fecha_hora" not in analisis["parametros_extraidos"]


def test_promociones_requieren_reglas_estructuradas_completas() -> None:
    promocion = {
        "activa": True,
        "descuento_max": 10,
        "condiciones": {
            "servicios": ["desarrollo_web"],
            "cantidad_minima": 10,
            "cantidad_maxima": 50,
            "canales": ["WhatsApp"],
        },
    }
    assert promocion_elegible(
        promocion, "desarrollo_web", 30, canal="WhatsApp"
    )
    assert not promocion_elegible(
        promocion, "desarrollo_web", 30, canal="Webchat"
    )
    assert not promocion_elegible(
        promocion, "desarrollo_web", 5, canal="WhatsApp"
    )
    assert not promocion_elegible(
        {"activa": True, "descuento_max": 10}, "desarrollo_web", 30
    )
    assert not promocion_elegible(
        {
            **promocion,
            "condiciones": {**promocion["condiciones"], "segmento_no_verificado": "x"},
        },
        "desarrollo_web",
        30,
        canal="WhatsApp",
    )


def test_fechas_imposibles_y_pasadas_no_se_aceptan() -> None:
    ahora = datetime(2026, 11, 23, 19, 16, tzinfo=ZoneInfo("America/Bogota"))
    assert _extraer_fecha_hora("31 de noviembre del 2026 a las 9am", ahora) is None
    assert _extraer_fecha_hora("31/11/2026 9:00", ahora) is None

    ayer = (datetime.now(ZoneInfo("America/Bogota")) - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    analisis = ModeloNLPVentas().analizar_mensaje(
        f"Quiero agendar cita para {ayer} 09:00"
    )
    assert analisis["parametros_extraidos"]["fecha_hora_pasada"] is True


def test_estado_aprobacion_cotizacion_es_persistente_y_por_cliente() -> None:
    repo = RepositorioSQLite(":memory:")
    perfiles = Informacion(
        perfiles=repo.listar_perfiles_clientes(),
        _on_save=repo.guardar_perfil_cliente,
    )
    perfiles.obtener_o_crear_perfil("cliente_con_cotizacion")
    perfiles.obtener_o_crear_perfil("cliente_sin_cotizacion")
    perfiles.registrar_cotizacion_calculada(
        "cliente_con_cotizacion",
        valor_total=1250000,
        complejidad=2,
        parametros_cotizacion={
            "tipo_servicio": "sesion_producto",
            "sesiones_requeridas": 1,
            "cantidad_productos": 10,
            "fecha_agendamiento_pendiente": "2030-01-02 11:00",
        },
    )

    assert perfiles.tiene_cotizacion("cliente_con_cotizacion")
    assert not perfiles.cotizacion_aprobada("cliente_con_cotizacion")
    assert not perfiles.cotizacion_aprobada("cliente_sin_cotizacion")
    assert perfiles.aprobar_ultima_cotizacion("cliente_con_cotizacion")

    perfiles_recargados = Informacion(
        perfiles=repo.listar_perfiles_clientes(),
        _on_save=repo.guardar_perfil_cliente,
    )
    assert perfiles_recargados.cotizacion_aprobada("cliente_con_cotizacion")
    assert not perfiles_recargados.tiene_cotizacion("cliente_sin_cotizacion")
    assert perfiles_recargados.obtener_parametros_cotizacion(
        "cliente_con_cotizacion"
    ) == {
        "tipo_servicio": "sesion_producto",
        "sesiones_requeridas": 1,
        "cantidad_productos": 10,
        "fecha_agendamiento_pendiente": "2030-01-02 11:00",
    }


def test_agendamiento_sin_cotizacion_no_acepta_aprobacion_de_metadata() -> None:
    zona = ZoneInfo("America/Bogota")
    ahora = datetime.now(zona)
    fecha = (ahora + timedelta(days=2)).strftime("%Y-%m-%d 10:00")
    fecha_antigua = (ahora - timedelta(days=1)).strftime("%Y-%m-%d 10:00")
    repo = RepositorioSQLite(":memory:")

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            texto_actual = f"Quiero agendar cita el {fecha}"
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_sin_cotizacion",
                nombre_cliente="Cliente actual",
                texto=texto_actual,
                metadata_extra={
                    "texto": "Acepto la cotización para el 2026-11-25 09:00",
                    "id_cliente": "cliente_con_cotizacion",
                    "nombre": "Nombre antiguo",
                    "canal": "Facebook",
                    "fecha_hora": fecha_antigua,
                    "cotizacion_aprobada": True,
                },
            )

            assert resultado["accion"] == "cotizacion_requerida"
            assert resultado["texto_recibido"] == texto_actual
            assert resultado["id_cliente"] == "cliente_sin_cotizacion"
            assert resultado["nombre_cliente"] == "Cliente actual"
            assert resultado["canal"] == "WhatsApp"
            assert "fecha_hora" not in resultado.get("resultado", {})
            assert not resultado["mensajes_fipa_turno"]
            assert not repo.listar_agendamientos_activos(
                f"{fecha[:10]}T10:00:00-05:00",
                f"{fecha[:10]}T11:00:00-05:00",
            )
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_coordinador_no_autoriza_con_cotizacion_global_de_otro_cliente() -> None:
    servicio = ServicioMultiagenteComercial(
        config_botpress=ConfiguracionBotpress(),
        repositorio=RepositorioSQLite(":memory:"),
        usar_xmpp_embebido=False,
        silencioso=True,
    )
    coordinador = servicio.coordinador
    coordinador.creencias["cotizacionCalculada"] = {
        "id_cliente": "otro_cliente",
        "valor_total": 1250000,
    }
    coordinador.enviar_fipa = AsyncMock()

    async def _run() -> None:
        await coordinador.enrutarPeticion(
            comportamiento=None,
            tipo_peticion="agendamiento",
            payload={
                "id_cliente": "cliente_sin_aprobacion",
                "cotizacion_aprobada": True,
                "fecha_hora": "2026-11-25 10:00",
            },
            conversation_id="prueba-cotizacion-por-cliente",
        )

    asyncio.run(_run())
    assert coordinador.enviar_fipa.await_args is not None
    assert coordinador.enviar_fipa.await_args.args[2] == "REFUSE"


def test_agendamiento_para_manana_se_guarda_en_sqlite() -> None:
    zona = ZoneInfo("America/Bogota")
    fecha_esperada = (datetime.now(zona).date() + timedelta(days=1)).isoformat()
    repo = RepositorioSQLite(":memory:")

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            cotizacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_fecha_relativa",
                nombre_cliente="Cliente fecha relativa",
                texto="Cotizar desarrollo web de 30 horas complejidad 2",
            )
            assert cotizacion["accion"] == "cotizacion_entregada"

            sin_aprobacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_fecha_relativa",
                nombre_cliente="Cliente fecha relativa",
                texto="Quiero agendar una cita para mañana a las 10:00",
            )
            assert sin_aprobacion["accion"] == "cotizacion_aprobacion_requerida"

            aprobacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_fecha_relativa",
                nombre_cliente="Cliente fecha relativa",
                texto="Acepto la cotización",
                metadata_extra={
                    "fecha_hora": f"{(datetime.now(zona).date() + timedelta(days=7)).isoformat()} 16:00"
                },
            )
            assert aprobacion["accion"] == "datos_reserva_pendientes"
            assert not repo.listar_agendamientos_activos(
                f"{fecha_esperada}T10:00:00-05:00",
                f"{fecha_esperada}T11:00:00-05:00",
            )
            resultado = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_fecha_relativa",
                nombre="Andrea Fecha",
            )
            assert resultado["accion"] == "cita_confirmada"
            assert resultado["resultado"]["fecha_hora"] == f"{fecha_esperada} 10:00"
            assert servicio.entorno.perfilesClientes.cotizacion_aprobada(
                "cliente_fecha_relativa"
            )
            filas = repo.listar_agendamientos_activos(
                f"{fecha_esperada}T10:00:00-05:00",
                f"{fecha_esperada}T11:00:00-05:00",
            )
            assert len(filas) == 1
            assert filas[0]["nombre_cliente"] == "Andrea Fecha"
            assert filas[0]["metadata"]["datos_cliente"]["cedula"] == "1020304050"
            assert filas[0]["metadata"]["datos_cliente"]["telefono"] == "+573001112233"
            assert all(
                "1020304050" not in str(mensaje["fipa_json"])
                for mensaje in resultado["mensajes_fipa_turno"]
            )
            perfil_final = servicio.entorno.perfilesClientes.perfiles[
                "cliente_fecha_relativa"
            ]
            assert "1020304050" not in str(perfil_final["historial_interacciones"])
            assert "1020304050" not in str(perfil_final["parametros_parciales"])
        finally:
            await servicio.detener()

    asyncio.run(_run())


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
            assert perfs_1a == ["INFORM", "INFORM", "INFORM", "INFORM"]
            assert res_completa["resultado"]["resultado"] == "cotizacion_entregada"
            assert res_completa["resultado"]["valor_total"] > 0
            assert "$" in res_completa["respuesta"]
            entrada_horas = {
                "tipo_servicio": "desarrollo_web",
                "cantidad_unidades": 45,
                "horas_estimadas": 45,
                "complejidad": 3,
            }
            assert res_completa["resultado"]["valor_total"] == servicio.modelo_rf.predecir_cotizacion(
                entrada_horas
            )["valor_total"]

            res_directa = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573001119900",
                nombre_cliente="Julian Perez",
                texto="Quiero cotizar una automatizacion_ia de 50 horas complejidad 4",
            )
            perfs_directa = _extraer_performativas(res_directa["mensajes_fipa_turno"])
            assert perfs_directa == ["REQUEST", "AGREE", "REQUEST", "AGREE", "INFORM", "INFORM"]
            assert res_directa["resultado"]["resultado"] == "cotizacion_entregada"

            res_no_viable = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="573001117766",
                nombre_cliente="Mario Diaz",
                texto="Quiero cotizar un servicio de plomeria de 20 horas",
            )
            perfs_no_viable = _extraer_performativas(res_no_viable["mensajes_fipa_turno"])
            assert perfs_no_viable == ["REQUEST", "REFUSE"]
            assert res_no_viable["resultado"]["resultado"] == "solicitud_no_viable"
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_negociacion_solo_aplica_el_descuento_autorizado() -> None:
    """
    Demuestra Escenario 2 (FIPA-Propose) en canal Webchat:
    - alt [acepta]: PROPOSE -> ACCEPT-PROPOSAL -> INFORM -> promocion presentada
    - una solicitud superior al límite se restringe a la promoción autorizada
    """

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            servicio.entorno.tarifasYMargenes.promociones_vigentes = [
                {
                    "codigo": "PROMO_ECOMMERCE_TEST",
                    "activa": True,
                    "descuento_max": 8,
                    "condiciones": {
                        "servicios": ["ecommerce"],
                        "cantidad_minima": 1,
                        "cantidad_maxima": 100,
                        "complejidad_minima": 1,
                        "complejidad_maxima": 5,
                        "canales": ["Webchat"],
                    },
                }
            ]
            cotizacion = await servicio.procesar_mensaje_entrante(
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
                "AGREE",
                "PROPOSE",
                "ACCEPT-PROPOSAL",
                "INFORM",
                "INFORM",
            ]
            assert res_2a["resultado"]["resultado"] == "promocion_presentada"
            precio_con_descuento = round(
                cotizacion["resultado"]["valor_total"] * 0.92, -2
            )
            assert f"${precio_con_descuento:,.0f}" in res_2a["respuesta"]

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
                "AGREE",
                "PROPOSE",
                "REJECT-PROPOSAL",
                "REFUSE",
            ] or perfs_2b == [
                "REQUEST",
                "REQUEST",
                "AGREE",
                "PROPOSE",
                "ACCEPT-PROPOSAL",
                "INFORM",
                "INFORM",
            ]
            assert res_2b["resultado"]["resultado"] == "promocion_presentada"
            assert "8%" in res_2b["respuesta"]
            assert "28%" not in res_2b["respuesta"]
            assert f"${precio_con_descuento:,.0f}" in res_2b["respuesta"]

            res_2c = await servicio.procesar_mensaje_entrante(
                canal="Webchat",
                id_cliente="web_user_88",
                nombre_cliente="Carlos Ruiz",
                texto="No voy a comprar, gracias, excede mi presupuesto",
            )
            perfs_2c = _extraer_performativas(res_2c["mensajes_fipa_turno"])
            assert perfs_2c == ["INFORM", "INFORM"]
            assert res_2c["resultado"]["resultado"] == "oportunidad_descartada"
            assert (
                servicio.entorno.perfilesClientes.perfiles["web_user_88"].get("estado_oportunidad")
                == "descartada"
            )
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_promocion_no_elegible_no_se_ofrece_y_se_mantiene_tarifa() -> None:
    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            servicio.entorno.tarifasYMargenes.promociones_vigentes = [
                {
                    "codigo": "PROMO_OTRO_SERVICIO",
                    "activa": True,
                    "descuento_max": 15,
                    "condiciones": {
                        "servicios": ["ecommerce"],
                        "cantidad_minima": 1,
                        "cantidad_maxima": 100,
                        "canales": ["WhatsApp"],
                    },
                }
            ]
            cotizacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_sin_promo",
                nombre_cliente="Cliente Promociones",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
                conversation_id="conv-sin-promo",
            )
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_sin_promo",
                nombre_cliente="Cliente Promociones",
                texto="El precio es demasiado caro",
                conversation_id="conv-sin-promo",
            )
            assert cotizacion["accion"] == "cotizacion_entregada"
            assert resultado["accion"] == "tarifa_base_mantenida"
            precio_normal = f"${cotizacion['resultado']['valor_total']:,.0f}"
            assert precio_normal in resultado["respuesta"]
            assert "PROMO" not in resultado["respuesta"]
            assert "PROPOSE" not in _extraer_performativas(
                resultado["mensajes_fipa_turno"]
            )
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_seleccion_de_horario_no_activa_promocion() -> None:
    fecha = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()
    repo = RepositorioSQLite(":memory:")
    repo.registrar_agendamiento(
        {
            "id_cliente": "cliente_slot_ocupado",
            "nombre_cliente": "Cliente Slot",
            "canal": "WhatsApp",
            "fecha_inicio": f"{fecha}T10:00:00-05:00",
            "fecha_fin": f"{fecha}T11:00:00-05:00",
            "estado": "confirmada",
            "registrado_en": "solo_supabase",
            "google_calendar_estado": "no_configurado",
        }
    )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            servicio.entorno.tarifasYMargenes.promociones_vigentes = [
                {
                    "codigo": "PROMO_DESARROLLO_TEST",
                    "activa": True,
                    "descuento_max": 8,
                    "condiciones": {
                        "servicios": ["desarrollo_web"],
                        "cantidad_minima": 1,
                        "cantidad_maxima": 100,
                        "complejidad_minima": 1,
                        "complejidad_maxima": 5,
                        "canales": ["WhatsApp"],
                    },
                }
            ]
            kwargs = {
                "canal": "WhatsApp",
                "id_cliente": "cliente_horario_sin_promo",
                "nombre_cliente": "Cliente Horario",
                "conversation_id": "conv-horario-sin-promo",
            }
            cotizacion = await servicio.procesar_mensaje_entrante(
                **kwargs,
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            solicitud = await servicio.procesar_mensaje_entrante(
                **kwargs,
                texto=f"Quiero agendar el {fecha} a las 10:00",
            )
            aceptacion = await servicio.procesar_mensaje_entrante(
                **kwargs, texto="Acepto la cotización"
            )
            alternativas = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_horario_sin_promo",
                conversation_id="conv-horario-sin-promo",
                nombre="Andrea Horario",
            )
            assert cotizacion["accion"] == "cotizacion_entregada"
            assert solicitud["accion"] == "cotizacion_aprobacion_requerida"
            assert aceptacion["accion"] == "datos_reserva_pendientes"
            assert alternativas["accion"] == "horarios_alternativos"
            opciones = alternativas["resultado"]["alternativas"]
            assert opciones
            hora = int(opciones[0][-5:-3])
            seleccion = await servicio.procesar_mensaje_entrante(
                **kwargs,
                texto=f"a las {hora} está bien",
            )
            assert seleccion["accion"] == "cita_confirmada"
            assert "descuento" not in seleccion["respuesta"].casefold()
            assert "PROPOSE" not in _extraer_performativas(
                seleccion["mensajes_fipa_turno"]
            )
            assert servicio.entorno.perfilesClientes.cotizacion_aprobada(
                "cliente_horario_sin_promo", "conv-horario-sin-promo"
            )
            assert repo.listar_agendamientos_activos(
                f"{fecha}T10:00:00-05:00", f"{fecha}T11:00:00-05:00"
            )
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_alternativas_rechazadas_no_se_vuelven_a_ofrecer() -> None:
    fecha = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=3)
    ).isoformat()
    repo = RepositorioSQLite(":memory:")
    for hora in (10, 11):
        repo.registrar_agendamiento(
            {
                "id_cliente": f"cliente_ocupado_{hora}",
                "nombre_cliente": f"Cliente Ocupado {hora}",
                "canal": "WhatsApp",
                "fecha_inicio": f"{fecha}T{hora:02}:00:00-05:00",
                "fecha_fin": f"{fecha}T{hora + 1:02}:00:00-05:00",
                "estado": "confirmada",
                "registrado_en": "solo_supabase",
                "google_calendar_estado": "no_configurado",
            }
        )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            kwargs = {
                "canal": "WhatsApp",
                "id_cliente": "cliente_rechaza_alt",
                "nombre_cliente": "Cliente Alternativas",
                "conversation_id": "conv-rechaza-alt",
            }
            await servicio.procesar_mensaje_entrante(
                **kwargs,
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            await servicio.procesar_mensaje_entrante(
                **kwargs,
                texto=f"Quiero agendar el {fecha} a las 10:00",
            )
            await servicio.procesar_mensaje_entrante(
                **kwargs, texto="Acepto la cotización"
            )
            primera = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_rechaza_alt",
                conversation_id="conv-rechaza-alt",
                nombre="Andrea Alternativas",
            )
            assert primera["accion"] == "horarios_alternativos"
            opciones_rechazadas = set(primera["resultado"]["alternativas"])
            assert opciones_rechazadas

            rechazo = await servicio.procesar_mensaje_entrante(
                **kwargs, texto="Ninguna de esas horas me conviene"
            )
            assert rechazo["accion"] == "horario_alternativo_rechazado"

            segunda = await servicio.procesar_mensaje_entrante(
                **kwargs, texto="a las 11 está bien"
            )
            assert segunda["accion"] == "horarios_alternativos"
            opciones_nuevas = set(segunda["resultado"]["alternativas"])
            assert opciones_nuevas
            assert opciones_nuevas.isdisjoint(opciones_rechazadas)
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_conversaciones_distintas_del_mismo_cliente_aislan_datos() -> None:
    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="botpress:user:estable",
                nombre_cliente="Cliente A",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
                conversation_id="conv-a",
            )
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="botpress:user:estable",
                nombre_cliente="Cliente A",
                texto="Acepto la cotización",
                conversation_id="conv-a",
            )
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="botpress:user:estable",
                nombre_cliente="Cliente A",
                texto="Ana Pérez",
                conversation_id="conv-a",
            )
            respuesta_b = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="botpress:user:estable",
                nombre_cliente="Cliente A",
                texto="Hola, ¿qué servicios ofrecen?",
                conversation_id="conv-b",
            )
            assert respuesta_b["accion"] == "catalogo_enviado"
            assert servicio.entorno.perfilesClientes.obtener_estado_conversacion(
                "botpress:user:estable", "conv-a"
            )["estado_reserva"]["paso"] == "cedula"
            assert not servicio.entorno.perfilesClientes.obtener_estado_conversacion(
                "botpress:user:estable", "conv-b"
            ).get("estado_reserva")
            assert not servicio.entorno.perfilesClientes.tiene_cotizacion(
                "botpress:user:estable", "conv-b"
            )
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_agendamiento_y_escalamiento_human_in_the_loop() -> None:
    """
    Valida Agendamiento FIPA-Query (bloque ocupado vs libre) y pausa
    Human-in-the-Loop cuando el cliente adopta tono hostil.
    """

    async def _run() -> None:
        fecha_agenda = (
            datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
        ).isoformat()
        repo = RepositorioSQLite(":memory:")
        repo.registrar_agendamiento(
            {
                "id_cliente": "cliente_bloqueado",
                "nombre_cliente": "Cliente bloqueado",
                "canal": "Facebook",
                "fecha_inicio": f"{fecha_agenda}T14:00:00-05:00",
                "fecha_fin": f"{fecha_agenda}T15:00:00-05:00",
                "estado": "confirmada",
                "registrado_en": "solo_supabase",
                "google_calendar_estado": "no_configurado",
            }
        )
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
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
            aprobacion = await servicio.procesar_mensaje_entrante(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto="Acepto la cotización",
            )
            assert aprobacion["accion"] == "datos_reserva_pendientes"
            siguiente_paso = await _capturar_datos_cliente_reserva(
                servicio, "Facebook", "fb_501", nombre="Andrea Torres"
            )
            assert siguiente_paso["accion"] == "fecha_agendamiento_requerida"

            res_ocupado = await servicio.procesar_mensaje_entrante(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto=f"Quiero agendar una cita para el {fecha_agenda} 14:00",
            )
            assert "QUERY-IF" in _extraer_performativas(res_ocupado["mensajes_fipa_turno"])
            assert res_ocupado["resultado"]["resultado"] == "horarios_alternativos"

            res_libre = await servicio.procesar_mensaje_entrante(
                canal="Facebook",
                id_cliente="fb_501",
                nombre_cliente="Andrea Torres",
                texto=f"Entonces quiero reservar cita el {fecha_agenda} 10:00",
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
            assert res_libre["resultado"]["registrado_en"] == "solo_supabase"
            assert (
                res_libre["resultado"]["google_calendar_estado"]
                == "no_configurado"
            )
            assert "guardada exitosamente" in res_libre["respuesta"]
            assert (
                servicio.entorno.agendaVirtual.bloques[
                    f"{fecha_agenda} 10:00"
                ]["estado"]
                == "OCUPADO"
            )
            registro = servicio.repositorio.listar_agendamientos_activos(
                f"{fecha_agenda}T10:00:00-05:00",
                f"{fecha_agenda}T11:00:00-05:00",
            )
            assert len(registro) == 1
            assert registro[0]["registrado_en"] == "solo_supabase"
            assert registro[0]["google_calendar_estado"] == "no_configurado"

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
            human_reactivation_api_key="clave_admin_reactivacion_2026",
            owner_whatsapp_number="573005551122",
        )
        servicio = ServicioMultiagenteComercial(
            config_botpress=config,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        perfil_anonimo_pausado = servicio.entorno.perfilesClientes.actualizar_perfil(
            "Anonymous User (User)",
            canal="Webchat",
            automatizacion_activa=False,
        )
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
                r_sin_identidad = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={"channel": "Webchat", "text": "Hola"},
                )
                assert r_sin_identidad.status_code == 400
                assert "identificador no genérico" in r_sin_identidad.json()["detail"]

                r_id_generico = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "channel": "Webchat",
                        "userId": "Anonymous User (User)",
                        "text": "Hola",
                    },
                )
                assert r_id_generico.status_code == 400

                r_fallback_conversacion = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "channel": "Webchat",
                        "userId": "Anonymous User (User)",
                        "conversationId": "edge-test-conversation-001",
                        "text": "Hola, que servicios ofrecen?",
                    },
                )
                assert r_fallback_conversacion.status_code == 200
                assert r_fallback_conversacion.json()["id_cliente"] == (
                    "botpress:webchat:conversation:edge-test-conversation-001"
                )
                assert r_fallback_conversacion.json()["alcance_identidad"] == "conversacion"
                assert perfil_anonimo_pausado["automatizacion_activa"] is False
                assert servicio.entorno.perfilesClientes.perfiles[
                    "Anonymous User (User)"
                ]["automatizacion_activa"] is False

                perfil_legacy = servicio.entorno.perfilesClientes.actualizar_perfil(
                    "legacy-user",
                    canal="Webchat",
                    ultima_cotizacion=123_000,
                )
                r_usuario_legacy = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "channel": "Webchat",
                        "userId": "legacy-user",
                        "conversationId": "legacy-chat-1",
                        "text": "Hola, que servicios ofrecen?",
                    },
                )
                assert r_usuario_legacy.status_code == 200
                assert r_usuario_legacy.json()["id_cliente"] == "legacy-user"
                assert r_usuario_legacy.json()["fuente_identidad"] == "userId_legacy"
                assert perfil_legacy["ultima_cotizacion"] == 123_000

                r_mismo_user_otro_canal = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "channel": "WhatsApp",
                        "userId": "legacy-user",
                        "conversationId": "legacy-chat-2",
                        "text": "Hola, que servicios ofrecen?",
                    },
                )
                assert r_mismo_user_otro_canal.status_code == 200
                assert r_mismo_user_otro_canal.json()["id_cliente"] == (
                    "botpress:whatsapp:user:legacy-user"
                )
                assert perfil_legacy["canal"] == "Webchat"

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

                parcial_otro_cliente = await servicio.procesar_mensaje_entrante(
                    canal="Webchat",
                    id_cliente="bp_web_otro",
                    nombre_cliente="Cliente con cotización pendiente",
                    texto="Quiero cotizar desarrollo web",
                )
                assert parcial_otro_cliente["accion"] == "parametros_incompletos"

                r_web = await client.post(
                    "/api/botpress/mensaje",
                    headers={"Authorization": "Bearer clave_botpress_2026"},
                    json={
                        "channel": "Webchat",
                        "userId": "bp_web_9001",
                        "conversationId": "webchat-turno-1",
                        "text": "Hola, que servicios ofrecen?",
                    },
                )
                assert r_web.status_code == 200
                data_web = r_web.json()
                assert data_web["canal"] == "Webchat"
                assert data_web["accion"] == "catalogo_enviado"
                assert len(data_web["respuesta"]) > 20
                id_web = "botpress:webchat:user:bp_web_9001"
                assert data_web["id_cliente"] == id_web
                perfil_web = servicio.entorno.perfilesClientes.perfiles[id_web]
                perfil_otro = servicio.entorno.perfilesClientes.perfiles["bp_web_otro"]
                assert not perfil_web["esperando_parametros_cotizacion"]
                assert not perfil_web["parametros_parciales"]
                assert perfil_otro["esperando_parametros_cotizacion"]

                r_web_turno_2 = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "channel": "Webchat",
                        "userId": "bp_web_9001",
                        "conversationId": "webchat-turno-2",
                        "text": "Quiero cotizar desarrollo web",
                    },
                )
                assert r_web_turno_2.status_code == 200
                assert r_web_turno_2.json()["id_cliente"] == id_web
                assert servicio.entorno.perfilesClientes.perfiles[id_web][
                    "esperando_parametros_cotizacion"
                ]

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

                r_fb_otro = await client.post(
                    "/api/botpress/mensaje",
                    headers=headers,
                    json={
                        "canal": "Facebook",
                        "id_cliente": "fb_7002",
                        "texto": "Esto es una estafa, necesito un humano",
                    },
                )
                assert r_fb_otro.status_code == 200
                assert r_fb_otro.json()["pausado_humano"] is True

                r_react_sin_clave = await client.post(
                    "/api/humano/reactivar",
                    json={"id_cliente": "fb_7001"},
                )
                assert r_react_sin_clave.status_code == 401
                assert servicio.entorno.perfilesClientes.perfiles["fb_7001"][
                    "automatizacion_activa"
                ] is False

                r_react_generico = await client.post(
                    "/api/humano/reactivar",
                    headers={"X-Admin-API-Key": "clave_admin_reactivacion_2026"},
                    json={"id_cliente": "Anonymous User (User)"},
                )
                assert r_react_generico.status_code == 400
                assert servicio.entorno.perfilesClientes.perfiles[
                    "Anonymous User (User)"
                ]["automatizacion_activa"] is False
                r_react_desconocido = await client.post(
                    "/api/humano/reactivar",
                    headers={"X-Admin-API-Key": "clave_admin_reactivacion_2026"},
                    json={"id_cliente": "perfil-inexistente"},
                )
                assert r_react_desconocido.status_code == 404
                assert "perfil-inexistente" not in (
                    servicio.entorno.perfilesClientes.perfiles
                )

                r_react = await client.post(
                    "/api/humano/reactivar",
                    headers={"X-Admin-API-Key": "clave_admin_reactivacion_2026"},
                    json={"id_cliente": "fb_7001"},
                )
                assert r_react.status_code == 200
                assert r_react.json()["perfil"]["automatizacion_activa"] is True
                assert servicio.entorno.perfilesClientes.perfiles["fb_7002"][
                    "automatizacion_activa"
                ] is False
                alertas_por_cliente = {
                    alerta["id_cliente"]: alerta["resuelta"]
                    for alerta in servicio.entorno.alertas_humanas
                }
                assert alertas_por_cliente["fb_7001"] is True
                assert alertas_por_cliente["fb_7002"] is False

                r_cal = await client.post(
                    "/api/calendario/disponibilidad",
                    json={"fecha_hora": "2026-11-19 09:00", "estado": "OCUPADO", "cliente": "VIP"},
                )
                assert r_cal.status_code == 200

                r_estado = await client.get(
                    "/api/estado",
                    headers={"X-Admin-API-Key": config.human_reactivation_api_key},
                )
                assert r_estado.status_code == 200
                estado = r_estado.json()
                assert len(estado["agentes"]) == 5
                assert len(estado["mensajes_fipa"]) > 0
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_estado_requiere_clave_admin_y_aisla_negocio() -> None:
    async def _run() -> None:
        config_a = ConfiguracionBotpress(
            human_reactivation_api_key="clave-admin-negocio-a",
            negocio_id="tech_freelance",
            sqlite_db_path=":memory:",
        )
        servicio_a = ServicioMultiagenteComercial(
            config_botpress=config_a,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        servicio_a.entorno.perfilesClientes.actualizar_perfil(
            "cliente-solo-a", nombre="Cliente A"
        )
        app_a = crear_app_botpress(servicio_a)

        config_b = ConfiguracionBotpress(
            human_reactivation_api_key="clave-admin-negocio-b",
            negocio_id="estudio_fotografia",
            sqlite_db_path=":memory:",
        )
        servicio_b = ServicioMultiagenteComercial(
            config_botpress=config_b,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        servicio_b.entorno.perfilesClientes.actualizar_perfil(
            "cliente-solo-b", nombre="Cliente B"
        )
        app_b = crear_app_botpress(servicio_b)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_a),
            base_url="http://testserver",
        ) as client_a, httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_b),
            base_url="http://testserver",
        ) as client_b:
            sin_clave = await client_a.get("/api/estado")
            clave_incorrecta = await client_a.get(
                "/api/estado",
                headers={"X-Admin-API-Key": "clave-incorrecta"},
            )
            estado_a = await client_a.get(
                "/api/estado",
                params={"negocio_id": "estudio_fotografia"},
                headers={"X-Admin-API-Key": config_a.human_reactivation_api_key},
            )
            clave_de_otro_negocio = await client_a.get(
                "/api/estado",
                headers={"X-Admin-API-Key": config_b.human_reactivation_api_key},
            )
            estado_b = await client_b.get(
                "/api/estado",
                headers={"X-Admin-API-Key": config_b.human_reactivation_api_key},
            )

        assert sin_clave.status_code == 401
        assert clave_incorrecta.status_code == 401
        assert "clave-incorrecta" not in clave_incorrecta.text
        assert estado_a.status_code == 200
        assert estado_a.json()["negocio"]["id"] == "tech_freelance"
        assert "cliente-solo-a" in {
            perfil["id_cliente"] for perfil in estado_a.json()["perfiles_clientes"]
        }
        assert "cliente-solo-b" not in {
            perfil["id_cliente"] for perfil in estado_a.json()["perfiles_clientes"]
        }
        assert clave_de_otro_negocio.status_code == 401
        assert estado_b.status_code == 200
        assert estado_b.json()["negocio"]["id"] == "estudio_fotografia"
        assert "cliente-solo-b" in {
            perfil["id_cliente"] for perfil in estado_b.json()["perfiles_clientes"]
        }
        assert "cliente-solo-a" not in {
            perfil["id_cliente"] for perfil in estado_b.json()["perfiles_clientes"]
        }

    asyncio.run(_run())


def test_estado_falla_cerrado_sin_clave_admin_aunque_haya_clave_botpress() -> None:
    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(
                botpress_api_key="clave-botpress-no-admin",
                sqlite_db_path=":memory:",
            ),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        app = crear_app_botpress(servicio)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            sin_clave_admin = await client.get("/api/estado")
            con_clave_botpress = await client.get(
                "/api/estado",
                headers={"X-API-Key": "clave-botpress-no-admin"},
            )

        assert sin_clave_admin.status_code == 503
        assert con_clave_botpress.status_code == 503
        assert "clave-botpress-no-admin" not in con_clave_botpress.text

    asyncio.run(_run())


def test_reactivacion_admin_falla_cerrada_sin_clave_configurada() -> None:
    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            perfil = servicio.entorno.perfilesClientes.actualizar_perfil(
                "cliente-pausado",
                automatizacion_activa=False,
            )
            app = crear_app_botpress(servicio)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                respuesta = await client.post(
                    "/api/humano/reactivar",
                    headers={"X-Admin-API-Key": "clave-no-configurada"},
                    json={"id_cliente": "cliente-pausado"},
                )
            assert respuesta.status_code == 503
            assert perfil["automatizacion_activa"] is False
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
            '["retrato", "corporativo", "marca personal"]',
            0,
            120_000.0,
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
            "sesion_producto",
            2,
            "Fotografia de Producto E-Commerce",
            "Iluminacion en estudio y retoque editorial",
            280_000.0,
            '["producto", "catalogo", "ecommerce", "e-commerce"]',
            1,
            150_000.0,
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
        cantidad = 1 + (i % 5)
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
                f"PROD-{i:03d}",
                "sesion_producto",
                2,
                1 + (i % 5),
                cantidad,
                10,
                i % 2,
                float(cantidad * 280_000),
                float(cantidad * 280_000 + (i % 2) * 150_000),
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
    conn.commit()

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
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
            assert res_cot["resultado"]["valor_total"] == 960_000

            res_incompleta = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_secuencia",
                nombre_cliente="Cliente prueba",
                texto="Necesito el servicio de Fotografía de Producto E-Commerce",
            )
            assert res_incompleta["accion"] == "parametros_incompletos"
            assert res_incompleta["resultado"]["faltantes"] == [
                "sesiones_requeridas",
                "complejidad",
            ]

            fecha_solicitada = (
                datetime.now(ZoneInfo("America/Bogota")) + timedelta(days=30)
            ).strftime("%Y-%m-%d 11:00")
            res_fecha = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_secuencia",
                nombre_cliente="Cliente prueba",
                texto=f"Quisiera agendar para el {fecha_solicitada}",
            )
            assert res_fecha["accion"] == "cotizacion_requerida"
            assert (
                servicio.entorno.perfilesClientes.obtener_parametros_cotizacion(
                    "cli_foto_secuencia"
                )["fecha_agendamiento_pendiente"]
                == fecha_solicitada
            )

            res_sesion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_secuencia",
                nombre_cliente="Cliente prueba",
                texto="Necesito una sola sesión y son 10 productos",
                metadata_extra={
                    "parametros_servicio": {
                        "cantidad_unidades": 9,
                        "sesiones_requeridas": 9,
                        "unidad_cantidad": "sesiones",
                    }
                },
            )
            assert res_sesion["accion"] == "parametros_incompletos"
            assert res_sesion["resultado"]["faltantes"] == ["complejidad"]
            parciales = servicio.entorno.perfilesClientes.obtener_parametros_cotizacion(
                "cli_foto_secuencia"
            )
            assert parciales["sesiones_requeridas"] == 1
            assert parciales["cantidad_productos"] == 10
            assert parciales["fecha_agendamiento_pendiente"] == fecha_solicitada

            res_complejidad = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_secuencia",
                nombre_cliente="Cliente prueba",
                texto="1",
                metadata_extra={"parametros_servicio": {"complejidad": 5}},
            )
            assert res_complejidad["accion"] == "cotizacion_entregada"
            assert res_complejidad["resultado"]["desglose"]["cantidad_unidades"] == 1
            assert res_complejidad["resultado"]["desglose"]["unidad_cantidad"] == "sesiones"
            assert res_complejidad["resultado"]["desglose"]["complejidad"] == 1
            assert res_complejidad["resultado"]["desglose"]["cantidad_productos"] == 10
            assert res_complejidad["resultado"]["valor_total"] == 280_000
            assert res_complejidad["resultado"]["desglose"]["tarifa_unitaria"] == 280_000
            assert res_complejidad["resultado"]["desglose"][
                "ajuste_complejidad_urgencia"
            ] == 0
            assert "no se aplica un recargo" in res_complejidad["respuesta"].lower()
            assert "1 sesión" in res_complejidad["respuesta"].lower()
            perfil_cotizado = servicio.entorno.perfilesClientes.perfiles[
                "cli_foto_secuencia"
            ]
            assert perfil_cotizado["parametros_parciales"][
                "_estado_cotizacion"
            ]["aprobada"] is False
            assert perfil_cotizado["parametros_parciales"][
                "fecha_agendamiento_pendiente"
            ] == fecha_solicitada
            assert not repo.listar_agendamientos_activos(
                f"{fecha_solicitada[:10]}T11:00:00-05:00",
                f"{fecha_solicitada[:10]}T12:00:00-05:00",
            )
            assert servicio.entorno.agendaVirtual.google_calendar is None
            assert servicio.entorno.agendaVirtual.verificar_disponibilidad(
                fecha_solicitada
            ), servicio.entorno.agendaVirtual.bloques

            res_rechazo = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_secuencia",
                nombre_cliente="Cliente prueba",
                texto="No acepto la cotización",
            )
            assert res_rechazo["accion"] == "cotizacion_aprobacion_requerida"
            assert not servicio.entorno.perfilesClientes.cotizacion_aprobada(
                "cli_foto_secuencia"
            )
            assert not repo.listar_agendamientos_activos(
                f"{fecha_solicitada[:10]}T11:00:00-05:00",
                f"{fecha_solicitada[:10]}T12:00:00-05:00",
            )

            aprobacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_secuencia",
                nombre_cliente="Cliente prueba",
                texto="Acepto la cotización",
            )
            assert aprobacion["accion"] == "datos_reserva_pendientes"
            res_aceptacion = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cli_foto_secuencia",
                nombre="Camila Torres",
            )
            assert res_aceptacion["accion"] == "cita_confirmada"
            assert res_aceptacion["resultado"]["fecha_hora"] == fecha_solicitada
            assert servicio.entorno.perfilesClientes.cotizacion_aprobada(
                "cli_foto_secuencia"
            )
            assert not servicio.entorno.perfilesClientes.obtener_parametros_cotizacion(
                "cli_foto_secuencia"
            ).get("fecha_agendamiento_pendiente")
            conv_ids = [
                resultado["mensajes_fipa_turno"][0]["fipa_json"][":conversation-id"]
                for resultado in (
                    res_incompleta,
                    res_sesion,
                    res_complejidad,
                    res_aceptacion,
                )
            ]
            assert len(set(conv_ids)) == len(conv_ids)

            otro_cliente = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_foto_diferente",
                nombre_cliente="Otro cliente",
                texto="Una sola sesión",
            )
            assert otro_cliente["accion"] == "parametros_incompletos"
            assert "tipo_servicio" in otro_cliente["resultado"]["faltantes"]
            parciales_otro = servicio.entorno.perfilesClientes.obtener_parametros_cotizacion(
                "cli_foto_diferente"
            )
            assert "tipo_servicio" not in parciales_otro
            assert parciales_otro["cantidad_unidades"] == 1
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
        supabase_url="https://demo-proyecto.supabase.co",
        supabase_key="service_role_demo",
        negocio_id="negocio_nube",
        http_client=client,
    )
    cfg = repo.obtener_configuracion_negocio()
    assert cfg["moneda"] == "USD"
    assert cfg["telefono_dueno"] == "573001119988"

    cat = repo.obtener_catalogo_servicios()
    assert "auditoria" in cat
    assert cat["auditoria"]["tarifa_unitaria"] == 120.0

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


def test_repositorio_supabase_crud_agendamientos() -> None:
    """Ejercita INSERT, consulta y UPDATE de citas por la API REST de Supabase."""
    import json

    filas: List[dict] = []
    payloads_enviados: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/agendamientos") and request.method == "GET":
            return httpx.Response(200, json=filas)
        if request.url.path.endswith("/agendamientos") and request.method == "POST":
            registro = json.loads(request.read().decode("utf-8"))
            payloads_enviados.append(registro)
            almacenado = {
                **registro,
                "fecha_inicio": datetime.fromisoformat(
                    registro["fecha_inicio"]
                ).astimezone(ZoneInfo("UTC")).isoformat(),
                "fecha_fin": datetime.fromisoformat(
                    registro["fecha_fin"]
                ).astimezone(ZoneInfo("UTC")).isoformat(),
            }
            filas.append(almacenado)
            return httpx.Response(201, json=[almacenado])
        if request.url.path.endswith("/agendamientos") and request.method == "PATCH":
            cambios = json.loads(request.read().decode("utf-8"))
            id_solicitado = request.url.params["id"].removeprefix("eq.")
            for fila in filas:
                if fila["id"] == id_solicitado:
                    fila.update(cambios)
                    return httpx.Response(200, json=[fila])
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    repo = RepositorioSupabase(
        supabase_url="https://demo-proyecto.supabase.co",
        supabase_key="service_role_demo",
        negocio_id="negocio_nube",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    registro = repo.registrar_agendamiento(
        {
            "id_cliente": "cliente_api",
            "nombre_cliente": "Cliente API",
            "canal": "Webchat",
            "fecha_inicio": "2026-11-20T14:00:00Z",
            "fecha_fin": "2026-11-20T15:00:00Z",
            "fecha_hora": "2026-11-20T09:00:00-05:00",
            "estado": "pendiente",
            "registrado_en": "solo_supabase",
            "google_calendar_estado": "error",
            "google_calendar_error": "No se pudo validar Google Calendar.",
        }
    )
    actualizada = repo.actualizar_agendamiento(
        registro["id"],
        {
            "estado": "confirmada",
            "google_calendar_estado": "no_configurado",
            "google_calendar_error": None,
        },
    )
    consulta = repo.listar_agendamientos_activos(
        "2026-11-20T09:00:00-05:00",
        "2026-11-20T10:00:00-05:00",
    )
    assert actualizada["estado"] == "confirmada"
    assert actualizada["google_calendar_estado"] == "no_configurado"
    assert len(consulta) == 1
    assert consulta[0]["id"] == registro["id"]
    assert registro["fecha_inicio"] == "2026-11-20T09:00:00-05:00"
    assert registro["fecha_fin"] == "2026-11-20T10:00:00-05:00"
    assert "fecha_hora" not in payloads_enviados[0]
    assert set(payloads_enviados[0]) == {
        "id",
        "negocio_id",
        "id_cliente",
        "nombre_cliente",
        "canal",
        "servicio",
        "fecha_inicio",
        "fecha_fin",
        "estado",
        "registrado_en",
        "google_calendar_estado",
        "google_event_id",
        "google_event_link",
        "google_calendar_error",
        "observaciones",
        "metadata",
    }
    assert payloads_enviados[0]["fecha_inicio"] == "2026-11-20T09:00:00-05:00"
    assert payloads_enviados[0]["fecha_fin"] == "2026-11-20T10:00:00-05:00"


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (409, {}),
        (400, {"code": "23P01", "message": "exclusion constraint"}),
    ],
)
def test_repositorio_supabase_traduce_conflictos_de_exclusion(
    status_code: int, body: dict
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(status_code, json=body)

    repo = RepositorioSupabase(
        supabase_url="https://demo-proyecto.supabase.co",
        supabase_key="service_role_demo",
        negocio_id="negocio_nube",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ConflictoAgendamientoError):
        repo.registrar_agendamiento(
            {
                "id_cliente": "cliente_conflicto",
                "nombre_cliente": "Cliente",
                "canal": "WhatsApp",
                "fecha_inicio": "2026-12-01T09:00:00-05:00",
                "fecha_fin": "2026-12-01T10:00:00-05:00",
                "estado": "confirmada",
                "registrado_en": "solo_supabase",
                "google_calendar_estado": "no_configurado",
            }
        )


def test_entorno_persiste_en_supabase_y_sincroniza_google_con_mocks() -> None:
    import json

    filas: List[dict] = []
    eventos: List[dict] = []
    consultas_freebusy: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/freeBusy"):
            consulta = json.loads(request.read().decode("utf-8"))
            consultas_freebusy.append(consulta)
            return httpx.Response(
                200, json={"calendars": {"owner@gmail.com": {"busy": []}}}
            )
        if request.method == "POST" and path.endswith("/events"):
            evento = json.loads(request.read().decode("utf-8"))
            eventos.append(evento)
            return httpx.Response(
                200, json={"id": "evt_supabase_mock", "htmlLink": "https://calendar.test/evt_supabase_mock"}
            )
        if path.endswith("/agendamientos") and request.method == "GET":
            return httpx.Response(200, json=filas)
        if path.endswith("/agendamientos") and request.method == "POST":
            fila = json.loads(request.read().decode("utf-8"))
            filas.append(fila)
            return httpx.Response(201, json=[fila])
        if path.endswith("/agendamientos") and request.method == "PATCH":
            cambios = json.loads(request.read().decode("utf-8"))
            id_solicitado = request.url.params["id"].removeprefix("eq.")
            for fila in filas:
                if fila["id"] == id_solicitado:
                    fila.update(cambios)
                    return httpx.Response(200, json=[fila])
            return httpx.Response(200, json=[])
        if path.endswith("/negocios"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[])

    repo = RepositorioSupabase(
        supabase_url="https://demo-proyecto.supabase.co",
        supabase_key="service_role_demo",
        negocio_id="negocio_nube",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    calendar = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        timezone="America/Bogota",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_estatico="token_gcal_demo",
    )
    entorno = EntornoVirtualAtencion(repositorio=repo, google_calendar=calendar)

    resultado = entorno.registrarCitaEnCalendario(
        fecha_hora="2026-12-01T14:00:00Z",
        cliente="Cliente Mock",
        id_cliente="cliente_supabase_mock",
        canal="WhatsApp",
        servicio="desarrollo_web",
        metadata={
            "datos_cliente": {
                "nombre_cliente": "Cliente Mock",
                "cedula": "1020304050",
                "telefono": "+573001112233",
            }
        },
    )

    assert resultado["status"] == "ok"
    assert resultado["estado"] == "confirmada"
    assert resultado["registrado_en"] == "supabase_y_google_calendar"
    assert resultado["google_calendar_estado"] == "sincronizado"
    assert resultado["google_event_id"] == "evt_supabase_mock"
    assert resultado["fecha_inicio"] == "2026-12-01T09:00:00-05:00"
    assert consultas_freebusy[0]["timeMin"] == "2026-12-01T09:00:00-05:00"
    assert eventos[0]["start"]["dateTime"] == "2026-12-01T09:00:00-05:00"
    fila = repo.obtener_agendamiento(resultado["id"])
    assert fila is not None
    assert fila["estado"] == "confirmada"


def test_entorno_persiste_en_supabase_sin_google() -> None:
    import json

    filas: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/agendamientos"):
            if request.method == "GET":
                return httpx.Response(200, json=filas)
            if request.method == "POST":
                fila = json.loads(request.read().decode("utf-8"))
                filas.append(fila)
                return httpx.Response(201, json=[fila])
        return httpx.Response(200, json=[])

    repo = RepositorioSupabase(
        supabase_url="https://demo-proyecto.supabase.co",
        supabase_key="service_role_demo",
        negocio_id="negocio_nube",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    entorno = EntornoVirtualAtencion(repositorio=repo)

    resultado = entorno.registrarCitaEnCalendario(
        fecha_hora="2026-12-01T09:00:00-05:00",
        cliente="Cliente Mock",
        id_cliente="cliente_supabase_sin_google",
        canal="WhatsApp",
        servicio="desarrollo_web",
        metadata={
            "datos_cliente": {
                "nombre_cliente": "Cliente Mock",
                "cedula": "1020304050",
                "telefono": "+573001112233",
            }
        },
    )

    assert resultado["status"] == "ok"
    assert resultado["estado"] == "confirmada"
    assert resultado["registrado_en"] == "solo_supabase"
    assert resultado["google_calendar_estado"] == "no_configurado"
    assert len(filas) == 1


def test_normaliza_fechas_iso_a_bogota_sin_perder_segundos() -> None:
    entorno = EntornoVirtualAtencion()
    datos = {
        "id_cliente": "cliente_fecha_iso",
        "nombre_cliente": "Cliente Fecha",
        "canal": "WhatsApp",
        "servicio": "desarrollo_web",
        "metadata": {
            "datos_cliente": {
                "nombre_cliente": "Cliente Fecha",
                "cedula": "1020304050",
                "telefono": "+573001112233",
            }
        },
    }

    utc = entorno.validarDatosAgendamiento(
        {**datos, "fecha_hora": "2026-11-20T14:00:00Z"}
    )
    assert utc["fecha_inicio"] == "2026-11-20T09:00:00-05:00"
    assert utc["fecha_hora"] == "2026-11-20 09:00"

    con_segundos = entorno.validarDatosAgendamiento(
        {**datos, "fecha_hora": "2026-11-20T09:00:30-05:00"}
    )
    assert con_segundos["fecha_inicio"] == "2026-11-20T09:00:30-05:00"
    assert con_segundos["fecha_fin"] == "2026-11-20T10:00:30-05:00"
    assert con_segundos["fecha_hora"] == "2026-11-20T09:00:30-05:00"

    calendar = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        token_estatico="token_gcal_demo",
    )
    inicio, fin = calendar._parsear_bloque("2026-11-20T14:00:30Z")
    assert inicio.isoformat() == "2026-11-20T09:00:30-05:00"
    assert fin.isoformat() == "2026-11-20T10:00:30-05:00"

    with pytest.raises(ValueError):
        entorno.validarDatosAgendamiento(
            {**datos, "fecha_hora": "fecha inválida"}
        )


def test_freebusy_informa_reason_sin_exponer_mensaje_del_calendario() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "calendars": {
                    "owner@gmail.com": {
                        "errors": [
                            {
                                "reason": "forbidden",
                                "message": "sensitive calendar details",
                            }
                        ]
                    }
                }
            },
        )

    calendar = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_estatico="token_gcal_demo",
    )
    inicio = datetime(2026, 11, 24, 9, 0, tzinfo=ZoneInfo("America/Bogota"))

    with pytest.raises(ErrorGoogleCalendar, match="forbidden") as error:
        calendar._consultar_freebusy(inicio, inicio + timedelta(minutes=30))

    assert "sensitive calendar details" not in str(error.value)


def test_reservas_canceladas_no_bloquean_y_pendientes_si() -> None:
    repo = RepositorioSQLite(":memory:")
    datos = {
        "id_cliente": "cliente_estado",
        "nombre_cliente": "Cliente",
        "canal": "WhatsApp",
        "registrado_en": "solo_supabase",
        "google_calendar_estado": "no_configurado",
    }
    repo.registrar_agendamiento(
        {
            **datos,
            "fecha_inicio": "2026-12-01T09:00:00-05:00",
            "fecha_fin": "2026-12-01T10:00:00-05:00",
            "estado": "cancelada",
        }
    )
    repo.registrar_agendamiento(
        {
            **datos,
            "fecha_inicio": "2026-12-01T10:00:00-05:00",
            "fecha_fin": "2026-12-01T11:00:00-05:00",
            "estado": "pendiente",
        }
    )
    entorno = EntornoVirtualAtencion(repositorio=repo)

    assert entorno.agendaVirtual.verificar_disponibilidad(
        "2026-12-01T09:00:00-05:00"
    )
    assert not entorno.agendaVirtual.verificar_disponibilidad(
        "2026-12-01T10:00:00-05:00"
    )
    repo.registrar_agendamiento(
        {
            **datos,
            "fecha_inicio": "2026-12-01T11:00:00-05:00",
            "fecha_fin": "2026-12-01T12:00:00-05:00",
            "estado": "confirmada",
        }
    )


def test_sqlite_permite_reservas_consecutivas_y_serializa_conflictos(
    tmp_path: Path,
) -> None:
    ruta_db = tmp_path / "concurrencia.sqlite"
    repositorios = [RepositorioSQLite(ruta_db), RepositorioSQLite(ruta_db)]
    base = {
        "nombre_cliente": "Cliente",
        "canal": "WhatsApp",
        "estado": "confirmada",
        "registrado_en": "solo_supabase",
        "google_calendar_estado": "no_configurado",
    }
    repositorios[0].registrar_agendamiento(
        {
            **base,
            "id_cliente": "cliente_1",
            "fecha_inicio": "2026-12-01T09:00:00-05:00",
            "fecha_fin": "2026-12-01T10:00:00-05:00",
        }
    )
    repositorios[0].registrar_agendamiento(
        {
            **base,
            "id_cliente": "cliente_2",
            "fecha_inicio": "2026-12-01T10:00:00-05:00",
            "fecha_fin": "2026-12-01T11:00:00-05:00",
        }
    )

    barrera = Barrier(2)

    def reservar(repo: RepositorioSQLite, cliente: str) -> str:
        barrera.wait()
        try:
            repo.registrar_agendamiento(
                {
                    **base,
                    "id_cliente": cliente,
                    "fecha_inicio": "2026-12-01T12:00:00-05:00",
                    "fecha_fin": "2026-12-01T13:00:00-05:00",
                }
            )
        except ConflictoAgendamientoError:
            return "conflicto"
        return "guardada"

    with ThreadPoolExecutor(max_workers=2) as pool:
        resultados = list(
            pool.map(
                lambda args: reservar(*args),
                [(repositorios[0], "cliente_3"), (repositorios[1], "cliente_4")],
            )
        )
    assert sorted(resultados) == ["conflicto", "guardada"]


def test_calendario_google_api_v3() -> None:
    """
    Valida AgenteAgendamientos (FIPA-Query) conectado al cliente Google Calendar API v3:
    1) Bloque ocupado en freeBusy -> propone alternativas libres reales del dia.
    2) Bloque libre -> crea el evento real mediante POST /calendars/{id}/events.
    """
    eventos_creados: List[dict] = []
    fecha_prueba = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/freeBusy"):
            body = request.read().decode("utf-8")
            # Primer consulta puntual sobre 14:00 -> ocupado; consulta jornada completa -> 09:00 libre
            if (
                f"{fecha_prueba}T14:00:00" in body
                and f"{fecha_prueba}T15:00:00" in body
            ):
                return httpx.Response(
                    200,
                    json={
                        "calendars": {
                            "owner@gmail.com": {
                                "busy": [
                                    {
                                        "start": f"{fecha_prueba}T14:00:00-05:00",
                                        "end": f"{fecha_prueba}T15:00:00-05:00",
                                    }
                                ]
                            }
                        }
                    },
                )
            if f"{fecha_prueba}T08:00:00" in body:
                return httpx.Response(
                    200,
                    json={
                        "calendars": {
                            "owner@gmail.com": {
                                "busy": [
                                    {
                                        "start": f"{fecha_prueba}T14:00:00-05:00",
                                        "end": f"{fecha_prueba}T15:00:00-05:00",
                                    }
                                ]
                            }
                        }
                    },
                )
            return httpx.Response(
                200, json={"calendars": {"owner@gmail.com": {"busy": []}}}
            )

        if request.method == "POST" and path.endswith("/events"):
            import json as _json

            payload = _json.loads(request.read().decode("utf-8"))
            eventos_creados.append(payload)
            return httpx.Response(
                200,
                json={"id": "evt_gcal_001", "htmlLink": "https://calendar.google.com/evt_gcal_001"},
            )

        if request.method == "GET" and path.endswith("/events"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "summary": "Cita Comercial - Laura",
                            "start": {
                                "dateTime": f"{fecha_prueba}T09:00:00-05:00"
                            },
                        }
                    ]
                },
            )
        return httpx.Response(404)

    gcal = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        timezone="America/Bogota",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_estatico="token_gcal_demo",
    )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            google_calendar=gcal,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            assert servicio.entorno.agendaVirtual.modo == "Google Calendar API v3"
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gcal",
                nombre_cliente="Laura",
                texto="Cotizar desarrollo web de 30 horas complejidad 2",
            )
            aprobacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gcal",
                nombre_cliente="Laura",
                texto="Acepto la cotización",
            )
            assert aprobacion["accion"] == "datos_reserva_pendientes"
            fecha_requerida = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cli_gcal",
                nombre="Laura Gomez",
            )
            assert fecha_requerida["accion"] == "fecha_agendamiento_requerida"
            res_ocup = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gcal",
                nombre_cliente="Laura",
                texto=f"Quiero agendar cita el {fecha_prueba} 14:00",
            )
            assert res_ocup["accion"] == "horarios_alternativos"
            assert f"{fecha_prueba} 09:00" in res_ocup["resultado"]["alternativas"]

            res_ok = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gcal",
                nombre_cliente="Laura",
                texto=f"Entonces agenda el {fecha_prueba} 09:00",
            )
            assert res_ok["accion"] == "cita_confirmada"
            assert len(eventos_creados) == 1
            assert "Laura Gomez" in eventos_creados[0]["summary"]
            assert "1020304050" not in str(eventos_creados[0])
            assert "+573001112233" in eventos_creados[0]["description"]
            assert eventos_creados[0]["start"]["dateTime"] == (
                f"{fecha_prueba}T09:00:00-05:00"
            )
            assert eventos_creados[0]["end"]["dateTime"] == (
                f"{fecha_prueba}T10:00:00-05:00"
            )
            registro = servicio.repositorio.listar_agendamientos_activos(
                f"{fecha_prueba}T09:00:00-05:00",
                f"{fecha_prueba}T10:00:00-05:00",
            )
            assert len(registro) == 1
            assert registro[0]["registrado_en"] == "supabase_y_google_calendar"
            assert registro[0]["google_calendar_estado"] == "sincronizado"
            assert registro[0]["google_event_id"] == "evt_gcal_001"
            assert registro[0]["google_event_link"].endswith("evt_gcal_001")
            assert "sincronizada con Google Calendar" in res_ok["respuesta"]
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_agendamiento_rechaza_horario_ocupado_en_persistencia() -> None:
    """Una cita activa persistida bloquea el slot y el agente ofrece alternativas."""
    fecha_agenda = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()
    repo = RepositorioSQLite(":memory:")
    repo.registrar_agendamiento(
        {
            "id_cliente": "cliente_previo",
            "nombre_cliente": "Cliente previo",
            "canal": "WhatsApp",
            "fecha_inicio": f"{fecha_agenda}T10:00:00-05:00",
            "fecha_fin": f"{fecha_agenda}T11:00:00-05:00",
            "estado": "confirmada",
            "registrado_en": "solo_supabase",
            "google_calendar_estado": "no_configurado",
        }
    )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_nuevo",
                nombre_cliente="Cliente nuevo",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            aprobacion = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_nuevo",
                nombre_cliente="Cliente nuevo",
                texto="Acepto la cotización",
            )
            assert aprobacion["accion"] == "datos_reserva_pendientes"
            fecha_requerida = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_nuevo",
                nombre="Cliente Nuevo",
            )
            assert fecha_requerida["accion"] == "fecha_agendamiento_requerida"
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_nuevo",
                nombre_cliente="Cliente nuevo",
                texto=f"Quiero agendar cita el {fecha_agenda} 10:00",
            )
            assert resultado["accion"] == "horarios_alternativos"
            assert resultado["resultado"]["alternativas"]
            assert len(
                repo.listar_agendamientos_activos(
                    f"{fecha_agenda}T10:00:00-05:00",
                    f"{fecha_agenda}T11:00:00-05:00",
                )
            ) == 1
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_error_de_supabase_no_confirma_cita() -> None:
    """Un fallo del INSERT se devuelve como FAILURE y nunca como cita confirmada."""
    fecha_prueba = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_falla_db",
                nombre_cliente="Cliente",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_falla_db",
                nombre_cliente="Cliente",
                texto="Acepto la cotización",
            )
            datos = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_falla_db",
                nombre="Cliente Falla",
            )
            assert datos["accion"] == "fecha_agendamiento_requerida"

            def falla_insert(_agendamiento: dict) -> dict:
                raise RuntimeError("fallo simulado al insertar en Supabase")

            setattr(
                servicio.repositorio,
                "registrar_agendamiento",
                falla_insert,
            )
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_falla_db",
                nombre_cliente="Cliente",
                texto=f"Quiero agendar cita el {fecha_prueba} 10:00",
            )
            performativas = _extraer_performativas(resultado["mensajes_fipa_turno"])
            assert "FAILURE" in performativas
            assert "CONFIRM" not in performativas
            assert resultado["accion"] == "error_agendamiento"
            assert "guardada exitosamente" not in resultado["respuesta"]
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_error_google_calendar_respalda_en_supabase_sin_declarar_sync() -> None:
    """Si freeBusy validó el slot pero events.insert falla, queda solo en Supabase."""
    fecha_prueba = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/freeBusy"):
            return httpx.Response(
                200, json={"calendars": {"owner@gmail.com": {"busy": []}}}
            )
        if request.method == "POST" and request.url.path.endswith("/events"):
            return httpx.Response(503, json={"error": "calendar temporarily unavailable"})
        return httpx.Response(404)

    calendar = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        timezone="America/Bogota",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_estatico="token_gcal_demo",
    )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            google_calendar=calendar,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_gcal_error",
                nombre_cliente="Cliente",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_gcal_error",
                nombre_cliente="Cliente",
                texto="Acepto la cotización",
            )
            datos = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_gcal_error",
                nombre="Cliente Error",
            )
            assert datos["accion"] == "fecha_agendamiento_requerida"
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_gcal_error",
                nombre_cliente="Cliente",
                texto=f"Quiero agendar cita el {fecha_prueba} 09:00",
            )
            fila = servicio.repositorio.listar_agendamientos_activos(
                f"{fecha_prueba}T09:00:00-05:00",
                f"{fecha_prueba}T10:00:00-05:00",
            )[0]
            assert resultado["accion"] == "cita_confirmada"
            assert fila["registrado_en"] == "solo_supabase"
            assert fila["google_calendar_estado"] == "error"
            assert fila["google_calendar_error"]
            assert resultado["resultado"]["google_calendar_estado"] == "error"
            assert "No se pudo sincronizar con Google Calendar" in resultado["respuesta"]
            assert "persistida en la base de datos" in resultado["respuesta"]
        finally:
            await servicio.detener()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "actualizacion_cometida_antes_del_error", [False, True]
)
def test_error_update_supabase_conserva_evento_para_reconciliacion(
    actualizacion_cometida_antes_del_error: bool,
) -> None:
    eventos_creados: List[dict] = []
    eventos_eliminados: List[str] = []
    fecha_prueba = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/freeBusy"):
            return httpx.Response(
                200, json={"calendars": {"owner@gmail.com": {"busy": []}}}
            )
        if request.method == "POST" and request.url.path.endswith("/events"):
            eventos_creados.append({})
            return httpx.Response(
                200,
                json={
                    "id": "evt_update_fallido",
                    "htmlLink": "https://calendar.google.com/evt_update_fallido",
                },
            )
        if request.method == "DELETE":
            eventos_eliminados.append(request.url.path)
            return httpx.Response(204)
        return httpx.Response(404)

    calendar = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        timezone="America/Bogota",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_estatico="token_gcal_demo",
    )
    repo = RepositorioSQLite(":memory:")

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            repositorio=repo,
            google_calendar=calendar,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_update_falla",
                nombre_cliente="Cliente",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_update_falla",
                nombre_cliente="Cliente",
                texto="Acepto la cotización",
            )
            datos = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_update_falla",
                nombre="Cliente Update",
            )
            assert datos["accion"] == "fecha_agendamiento_requerida"

            actualizar_original = repo.actualizar_agendamiento

            def falla_actualizacion_sync(
                id_agendamiento: str, cambios: dict
            ) -> dict:
                if cambios.get("google_calendar_estado") == "sincronizado":
                    if actualizacion_cometida_antes_del_error:
                        actualizar_original(id_agendamiento, cambios)
                    raise RuntimeError("fallo simulado al actualizar Supabase")
                return actualizar_original(id_agendamiento, cambios)

            repo.actualizar_agendamiento = falla_actualizacion_sync
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_update_falla",
                nombre_cliente="Cliente",
                texto=f"Quiero agendar cita el {fecha_prueba} 09:00",
            )
            filas = repo.listar_agendamientos_activos(
                f"{fecha_prueba}T09:00:00-05:00",
                f"{fecha_prueba}T10:00:00-05:00",
            )

            assert len(filas) == 1
            assert filas[0]["google_event_id"] == "evt_update_fallido"
            assert len(eventos_creados) == 1
            assert eventos_eliminados == []
            if actualizacion_cometida_antes_del_error:
                assert resultado["accion"] == "cita_confirmada"
                assert filas[0]["estado"] == "confirmada"
                assert filas[0]["google_calendar_estado"] == "sincronizado"
            else:
                assert resultado["accion"] == "agendamiento_pendiente"
                assert filas[0]["estado"] == "pendiente"
                assert filas[0]["google_calendar_estado"] == "error"
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_error_google_disponibilidad_deja_cita_pendiente_sin_confirmar() -> None:
    """Un freeBusy fallido deja una solicitud durable pendiente y sin CONFIRM."""
    fecha_prueba = (
        datetime.now(ZoneInfo("America/Bogota")).date() + timedelta(days=2)
    ).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/freeBusy"):
            return httpx.Response(503, json={"error": "calendar unavailable"})
        return httpx.Response(404)

    calendar = ClienteGoogleCalendar(
        calendar_id="owner@gmail.com",
        timezone="America/Bogota",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        token_estatico="token_gcal_demo",
    )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            google_calendar=calendar,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_gcal_pendiente",
                nombre_cliente="Cliente",
                texto="Cotizar desarrollo web de 30 horas y complejidad 2",
            )
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_gcal_pendiente",
                nombre_cliente="Cliente",
                texto="Acepto la cotización",
            )
            datos = await _capturar_datos_cliente_reserva(
                servicio,
                "WhatsApp",
                "cliente_gcal_pendiente",
                nombre="Cliente Pendiente",
            )
            assert datos["accion"] == "fecha_agendamiento_requerida"
            resultado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cliente_gcal_pendiente",
                nombre_cliente="Cliente",
                texto=f"Quiero agendar cita el {fecha_prueba} 09:00",
            )
            performativas = _extraer_performativas(
                resultado["mensajes_fipa_turno"]
            )
            filas = servicio.repositorio.listar_agendamientos_activos(
                f"{fecha_prueba}T09:00:00-05:00",
                f"{fecha_prueba}T10:00:00-05:00",
            )
            assert "FAILURE" in performativas
            assert "CONFIRM" not in performativas
            assert resultado["accion"] == "agendamiento_pendiente"
            assert len(filas) == 1
            assert filas[0]["estado"] == "pendiente"
            assert filas[0]["google_calendar_estado"] == "error"
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_agendamientos_sqlite_persisten_tras_reiniciar_repositorio(
    tmp_path: Path,
) -> None:
    """El respaldo SQLite no depende del diccionario en memoria del calendario."""
    ruta_db = tmp_path / "agenda.sqlite"
    repo = RepositorioSQLite(ruta_db)
    repo.registrar_agendamiento(
        {
            "id_cliente": "cliente_persistente",
            "nombre_cliente": "Cliente persistente",
            "canal": "Webchat",
            "fecha_inicio": "2026-11-20T09:00:00-05:00",
            "fecha_fin": "2026-11-20T10:00:00-05:00",
            "estado": "confirmada",
            "registrado_en": "solo_supabase",
            "google_calendar_estado": "no_configurado",
        }
    )

    repo_reiniciado = RepositorioSQLite(ruta_db)
    filas = repo_reiniciado.listar_agendamientos_activos(
        "2026-11-20T09:00:00-05:00",
        "2026-11-20T10:00:00-05:00",
    )
    assert len(filas) == 1
    assert filas[0]["id_cliente"] == "cliente_persistente"




def test_objecion_precio_sin_cotizacion_no_lanza_nameerror() -> None:
    """Regresion: objetar precio antes de cotizar responde limpiamente sin NameError."""

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            res = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_sin_cot_1",
                nombre_cliente="Sofia Morales",
                texto="Esta muy caro para mi presupuesto, tienen descuento?",
            )
            assert res["accion"] == "cotizacion_requerida"
            assert "cotización" in res["respuesta"].lower()
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_promociones_default_bd_son_elegibles_y_activan_knn() -> None:
    """Las promociones sembradas por defecto incluyen condiciones estructuradas validas."""

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_promo_default",
                nombre_cliente="Andres Castro",
                texto="Quiero cotizar un desarrollo web de 40 horas complejidad 3",
            )
            res_promo = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_promo_default",
                nombre_cliente="Andres Castro",
                texto="El precio esta costoso para mi presupuesto, tienen alguna rebaja?",
            )
            perfs = _extraer_performativas(res_promo["mensajes_fipa_turno"])
            assert "AGREE" in perfs
            assert "PROPOSE" in perfs
            assert "ACCEPT-PROPOSAL" in perfs
            assert res_promo["accion"] == "promocion_presentada"
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_coordinador_rechaza_descuento_que_compromete_margen_minimo() -> None:
    """Verifica la rama REJECT-PROPOSAL -> REFUSE cuando el descuento compromete rentabilidad."""

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            servicio.entorno.tarifasYMargenes.margen_minimo_ganancia = 0.34
            await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_margen_estricto",
                nombre_cliente="Diana Ruiz",
                texto="Quiero cotizar un ecommerce de 50 horas complejidad 4",
            )
            res = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_margen_estricto",
                nombre_cliente="Diana Ruiz",
                texto="Esta muy costoso, necesito un descuento",
            )
            perfs = _extraer_performativas(res["mensajes_fipa_turno"])
            assert perfs == [
                "REQUEST",
                "REQUEST",
                "AGREE",
                "PROPOSE",
                "REJECT-PROPOSAL",
                "REFUSE",
            ]
            assert res["accion"] == "descuento_rechazado"
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_generador_gemini_redacta_valida_hechos_y_hace_fallback_en_429() -> None:
    """
    Valida la arquitectura dual Scikit-Learn (NLU) + Gemini (NLG):
    1) Redacta respuesta humana conservando el valor exacto del RandomForest.
    2) Rechaza salida alucinada si Gemini altera el precio y cae al texto verificado.
    3) Activa circuit breaker ante HTTP 429 (cuota agotada) evitando bloquear futuras llamadas.
    """
    llamadas_http: List[str] = []
    modo_mock = {"caso": "ok"}

    def handler(request: httpx.Request) -> httpx.Response:
        llamadas_http.append(str(request.url))
        assert request.headers.get("x-goog-api-key") == "clave_gemini_demo"
        if modo_mock["caso"] == "429":
            return httpx.Response(429, json={"error": {"message": "Quota exceeded"}})
        if modo_mock["caso"] == "alucinacion":
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Claro que si, tu desarrollo web cuesta solo $500,000 COP."
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        body = request.read().decode("utf-8")
        # Extrae el monto exacto enviado en el prompt para redactar una respuesta natural valida
        import re as _re

        montos = _re.findall(r"\$[\d,]+", body)
        monto_real = montos[0] if montos else "$0"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "Con mucho gusto revise los detalles de tu proyecto de "
                                        "Desarrollo Web Corporativo / Landing & Portal. "
                                        f"La inversion estimada queda en {monto_real} COP. "
                                        "Si te parece bien, podemos agendar una reunion para arrancar."
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    tiempo_simulado = {"t": 100.0}
    generador = GeneradorDialogoGemini(
        api_key="clave_gemini_demo",
        modelo="gemini-2.5-flash",
        timeout_segundos=2.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        reloj=lambda: tiempo_simulado["t"],
    )

    async def _run() -> None:
        servicio = ServicioMultiagenteComercial(
            config_botpress=ConfiguracionBotpress(),
            generador_llm=generador,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            # 1) Redaccion natural exitosa conservando el valor del Random Forest
            res_ok = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gem_1",
                nombre_cliente="Mateo Silva",
                texto="Quiero cotizar un desarrollo web de 40 horas complejidad 3",
            )
            assert res_ok["accion"] == "cotizacion_entregada"
            assert "Con mucho gusto revise los detalles" in res_ok["respuesta"]
            assert f"${res_ok['resultado']['valor_total']:,.0f}" in res_ok["respuesta"]
            assert generador.llamadas_exitosas == 1
            assert len(llamadas_http) == 1

            # 2) Anti-alucinacion: si Gemini cambia el monto, cae al borrador verificado
            modo_mock["caso"] = "alucinacion"
            res_guard = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gem_2",
                nombre_cliente="Lucia Vega",
                texto="Quiero cotizar un desarrollo web de 40 horas complejidad 3",
            )
            assert "$500,000" not in res_guard["respuesta"]
            assert f"${res_guard['resultado']['valor_total']:,.0f}" in res_guard["respuesta"]
            assert generador.fallbacks_activados == 1

            # 3) HTTP 429 (tokens agotados): activa circuit breaker y no vuelve a llamar HTTP
            modo_mock["caso"] = "429"
            res_429 = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gem_3",
                nombre_cliente="Paula Ortiz",
                texto="Hola, que servicios ofrecen?",
            )
            assert res_429["accion"] == "catalogo_enviado"
            assert "complejidad 1-5" not in res_429["respuesta"].lower()
            assert generador.en_enfriamiento is True
            llamadas_tras_429 = len(llamadas_http)

            # Siguiente mensaje en enfriamiento responde al instante sin llamada HTTP
            res_enfriado = await servicio.procesar_mensaje_entrante(
                canal="WhatsApp",
                id_cliente="cli_gem_4",
                nombre_cliente="Oscar Leon",
                texto="Oye compadre que vendes",
            )
            assert res_enfriado["accion"] == "catalogo_enviado"
            assert len(llamadas_http) == llamadas_tras_429
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_cotizacion_fotografia_a_medida_fraccionada_sin_perdida() -> None:
    """
    Verifica cotizacion autonoma de servicios fotograficos fuera del catalogo
    (ej. foto familiar para 4 personas) y paquetes fraccionados (ej. solo 1 foto),
    protegiendo siempre el margen minimo de rentabilidad en AgenteCoordinador.
    """
    from src.reglas_comerciales import calcular_cotizacion_a_medida_fotografia

    una_foto = calcular_cotizacion_a_medida_fotografia(
        clave_servicio_ref="retrato_corporativo",
        nombre_paquete_ref="Retrato Corporativo y Marca Personal",
        tarifa_paquete_ref=320_000,
        descripcion_paquete_ref="Incluye 8 fotos editadas en alta resolucion en estudio.",
        descripcion_cliente="Fotografía Individual (1 foto)",
        margen_base=0.40,
        margen_minimo=0.25,
        cantidad_fotos=1,
        cantidad_personas=1,
        en_exterior=False,
    )
    desg_1 = una_foto["desglose"]
    assert 100_000 <= una_foto["valor_total"] < 320_000
    assert desg_1["margen_neto_estimado"] >= 0.25
    assert una_foto["valor_total"] >= desg_1["piso_minimo_rentable"]

    familiar_4 = calcular_cotizacion_a_medida_fotografia(
        clave_servicio_ref="retrato_corporativo",
        nombre_paquete_ref="Retrato Corporativo y Marca Personal",
        tarifa_paquete_ref=320_000,
        descripcion_paquete_ref="Incluye 8 fotos editadas en alta resolucion en estudio.",
        descripcion_cliente="Sesión de Fotografía Familiar",
        margen_base=0.40,
        margen_minimo=0.25,
        cantidad_fotos=8,
        cantidad_personas=4,
        en_exterior=False,
    )
    assert familiar_4["valor_total"] > una_foto["valor_total"]
    assert familiar_4["desglose"]["margen_neto_estimado"] >= 0.25

    async def _run() -> None:
        repo = RepositorioSQLite(ruta_db=":memory:", negocio_id="estudio_fotografia", auto_seed=False)
        repo._conn.execute(
            """
            INSERT INTO negocios (
                id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
                etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
                descuento_maximo, mensaje_catalogo_intro
            ) VALUES (
                'estudio_fotografia', 'Estudio Visual Independiente',
                'Fotografia Comercial y Produccion de Video', 'COP', 'sesiones',
                'sesiones_requeridas', 'complejidad', '573001234567', 0.40, 0.25,
                12.0, 'Hola! Aqui tienes nuestros paquetes'
            )
            """
        )
        repo._conn.execute(
            """
            INSERT INTO servicios (
                negocio_id, clave, servicio_id_num, nombre, descripcion, tarifa_unitaria,
                palabras_clave, requiere_complejidad, costo_adicional_mantenimiento
            ) VALUES
                ('estudio_fotografia', 'sesion_producto', 1, 'Fotografia de Producto E-Commerce', 'Incluye 10 fotos editadas en estudio.', 280000, '["producto","ecommerce"]', 1, 150000),
                ('estudio_fotografia', 'retrato_corporativo', 2, 'Retrato Corporativo y Marca Personal', 'Incluye 8 fotos editadas en estudio.', 320000, '["retrato","corporativo","foto"]', 0, 120000)
            """
        )
        repo._conn.execute(
            """
            INSERT INTO tarifas_historicas (
                negocio_id, id_cotizacion, tipo_servicio, tipo_servicio_id, complejidad,
                cantidad_unidades, urgencia_dias, incluye_adicional, costo_base, costo_total, margen_estimado
            ) VALUES
                ('estudio_fotografia', 'FOT-1', 'retrato_corporativo', 2, 2, 1, 5, 0, 320000, 320000, 0.40),
                ('estudio_fotografia', 'FOT-2', 'sesion_producto', 1, 3, 2, 7, 0, 560000, 560000, 0.40)
            """
        )
        repo._conn.execute(
            """
            INSERT INTO historico_ofertas (
                negocio_id, id_oferta, monto_cotizado, nivel_indecision, complejidad,
                fidelidad_cliente, porcentaje_descuento_optimo, cerrado_exitosamente
            ) VALUES
                ('estudio_fotografia', 'OFE-1', 320000, 0.7, 2, 1, 8.0, 1),
                ('estudio_fotografia', 'OFE-2', 560000, 0.8, 3, 2, 10.0, 1)
            """
        )
        repo._conn.commit()

        servicio = ServicioMultiagenteComercial(
            repositorio=repo,
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await servicio.iniciar()
        try:
            # 1) Servicio fuera de catalogo: foto familiar para 4 personas
            res_fam = await servicio.procesar_mensaje_entrante(
                canal="Webchat",
                id_cliente="cli_fam_1",
                nombre_cliente="Camila Diaz",
                texto="Quiero una foto familiar, somos 4 personas",
            )
            assert res_fam["intencion"] == "solicitud_cotizacion"
            assert res_fam["resultado"]["resultado"] == "cotizacion_entregada"
            desg_fam = res_fam["resultado"]["desglose"]
            assert desg_fam["es_personalizado"] is True
            assert desg_fam["cantidad_personas"] == 4
            assert desg_fam["margen_neto_estimado"] >= 0.25
            assert "aprobada_sin_perdida" in desg_fam["validacion_coordinador"]

            # 2) Servicio fraccionado: solo 1 foto de producto (no vende a perdida)
            res_1foto = await servicio.procesar_mensaje_entrante(
                canal="Webchat",
                id_cliente="cli_1foto",
                nombre_cliente="Luis Perez",
                texto="De fotografia de producto solo quiero 1 foto",
            )
            assert res_1foto["intencion"] == "solicitud_cotizacion"
            assert res_1foto["resultado"]["resultado"] == "cotizacion_entregada"
            val_1 = res_1foto["resultado"]["valor_total"]
            desg_prod = res_1foto["resultado"]["desglose"]
            assert desg_prod["cantidad_fotos"] == 1
            assert 90_000 <= val_1 < 280_000
            assert desg_prod["margen_neto_estimado"] >= 0.25

            # 3) Coordinador ajusta automaticamente si llegara una propuesta por debajo del piso
            payload_bajo = {
                "valor_total": 40_000.0,
                "desglose": {
                    "costo_operativo_directo": 75_000.0,
                    "piso_minimo_rentable": 100_000.0,
                },
            }
            validacion = servicio.coordinador.validarPropuestaComercial(payload_bajo)
            assert validacion["ajustada_por_margen"] is True
            assert payload_bajo["valor_total"] == 100_000.0
            assert payload_bajo["desglose"]["margen_neto_estimado"] >= 0.25
        finally:
            await servicio.detener()

    asyncio.run(_run())


def test_bateria_exhaustiva_escenarios_nlp_y_reglas() -> None:
    """
    Bateria exhaustiva de 50+ escenarios reales (coloquial colombiano, typos,
    fotografia estandar/a medida/fraccionada, software, objeciones de precio,
    fechas naturales, rechazos corteses, escalamiento humano y fuera de alcance)
    junto con validacion cruzada del clasificador TF-IDF + LogisticRegression.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from src.ml.datasets import CORPUS_NLP_VENTAS

    # 1) Validacion cruzada del modelo puro Scikit-Learn (sin reglas)
    modelo_tech = ModeloNLPVentas(unidad_cantidad="horas")
    textos = [fila[0] for fila in CORPUS_NLP_VENTAS]
    etiquetas = [fila[1] for fila in CORPUS_NLP_VENTAS]
    assert len(CORPUS_NLP_VENTAS) >= 150
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(modelo_tech.modelo_intencion, textos, etiquetas, cv=cv)
    assert float(scores.mean()) >= 0.80, f"Accuracy CV insuficiente: {scores.mean():.3f}"

    catalogo_foto = {
        "sesion_producto": {
            "id": 1,
            "nombre": "Fotografia de Producto E-Commerce",
            "tarifa_unitaria": 280_000.0,
            "palabras_clave": ["producto", "catalogo", "ecommerce"],
            "requiere_complejidad": True,
        },
        "retrato_corporativo": {
            "id": 2,
            "nombre": "Retrato Corporativo y Marca Personal",
            "tarifa_unitaria": 320_000.0,
            "palabras_clave": ["retrato", "corporativo", "perfil", "marca personal", "foto"],
            "requiere_complejidad": False,
        },
        "cobertura_evento": {
            "id": 3,
            "nombre": "Cobertura Fotografica de Eventos",
            "tarifa_unitaria": 450_000.0,
            "palabras_clave": ["evento", "boda", "grado", "conferencia", "fiesta"],
            "requiere_complejidad": True,
        },
        "video_reel": {
            "id": 4,
            "nombre": "Produccion de Video Reels / Comercial Corto",
            "tarifa_unitaria": 390_000.0,
            "palabras_clave": ["video", "reel", "reels", "comercial", "grabacion"],
            "requiere_complejidad": True,
        },
    }
    modelo_foto = ModeloNLPVentas(
        catalogo_servicios=catalogo_foto,
        unidad_cantidad="sesiones",
        etiqueta_cantidad="sesiones_requeridas",
    )

    casos_foto = [
        # Saludos y consultas
        ("Hola", "consulta_servicio"),
        ("Buenos días", "consulta_servicio"),
        ("Buenas noches", "consulta_servicio"),
        ("Me envías el catálogo por favor", "consulta_servicio"),
        ("Qué paquetes tienen disponibles?", "consulta_servicio"),
        ("Oye compadre qué vendes", "consulta_servicio"),
        ("Q servicios manejan?", "consulta_servicio"),
        # Cotizaciones estandar, a medida, fraccionadas y refinamientos
        ("Quiero cotizar 3 sesiones de retrato corporativo", "solicitud_cotizacion"),
        ("Vale, me puedes hacer una cotización?", "solicitud_cotizacion"),
        ("Quiero una foto familiar, somos 4 personas", "solicitud_cotizacion"),
        ("De fotografía de producto solo quiero 1 foto", "solicitud_cotizacion"),
        ("Pero yo nada más necesito 3 fotos", "solicitud_cotizacion"),
        ("Te cuento, sería una sesión, unas 7 fotos con mi esposa", "solicitud_cotizacion"),
        ("pero sería en el centro donde es la boda", "solicitud_cotizacion"),
        ("Y si es en exterior en la playa?", "solicitud_cotizacion"),
        ("Necesito fotos de 15 años en una finca", "solicitud_cotizacion"),
        ("Quiero presupuesto para fotos de comida de mi restaurante", "solicitud_cotizacion"),
        ("Cuánto vale una sesión de pareja?", "solicitud_cotizacion"),
        ("Solo necesito una foto para hoja de vida", "solicitud_cotizacion"),
        ("Necesito 10 fotos de un apartamento para Airbnb", "solicitud_cotizacion"),
        ("Quiero cotizar una sesión de fotos con mi mascota", "solicitud_cotizacion"),
        # Objeciones de precio (coloquiales y formales)
        ("Está muy caro para mi presupuesto", "objecion_precio"),
        ("Uy está muy cariñoso ese precio", "objecion_precio"),
        ("Me puedes hacer un descuentico?", "objecion_precio"),
        ("No me alcanza la plata, hay rebaja?", "objecion_precio"),
        ("Bájale un poquito al precio y cerramos", "objecion_precio"),
        ("En otro estudio me cobran más barato", "objecion_precio"),
        ("No tienes algo más económico?", "objecion_precio"),
        ("Xq tan caro?", "objecion_precio"),
        ("Dame un 10% de descuento", "objecion_precio"),
        ("Necesito un 25% de descuento porque es demasiado", "objecion_precio"),
        # Agendamiento y fechas naturales
        ("Acepto la cotización, agendemos para el 30 a las 3pm", "solicitud_agendamiento"),
        ("Listo, cerramos trato, separa el cupo", "solicitud_agendamiento"),
        ("Quiero reservar cita para mañana al mediodía", "solicitud_agendamiento"),
        ("Qué tal el sábado a las 10 de la mañana?", "solicitud_agendamiento"),
        ("El 5 de diciembre a las 4pm me sirve para agendar", "solicitud_agendamiento"),
        # Rechazo de compra (Alternativa C)
        ("No voy a comprar, gracias", "rechazo_compra"),
        ("Por ahora no gracias, mejor después", "rechazo_compra"),
        ("Deja así, cancelo la cotización", "rechazo_compra"),
        ("Declino la propuesta, no deseo continuar", "rechazo_compra"),
        ("No me convence, paso por ahora", "rechazo_compra"),
        # Escalamiento humano y reclamos criticos (Escenario 2)
        ("Esto es una estafa, pésimo servicio", "reclamo_critico"),
        ("Quiero hablar con un humano por favor", "reclamo_critico"),
        ("Pásame con el dueño del negocio", "reclamo_critico"),
        ("Prefiero hablar con una persona real", "reclamo_critico"),
        ("Este bot no sirve, necesito atención humana", "reclamo_critico"),
        ("Me robaron, exijo devolución inmediata", "reclamo_critico"),
        ("Voy a poner una queja ante la superintendencia", "reclamo_critico"),
    ]

    errores = []
    for texto, esperada in casos_foto:
        res = modelo_foto.analizar_mensaje(texto)
        if res["intencion"] != esperada:
            errores.append(f"'{texto}' -> obtuvo '{res['intencion']}', esperaba '{esperada}'")
    assert not errores, "Fallos de clasificacion:\n" + "\n".join(errores)

    # Verificar extracciones clave
    res_esposa = modelo_foto.analizar_mensaje("unas 7 fotos con mi esposa en el centro")
    assert res_esposa["parametros_extraidos"]["cantidad_fotos"] == 7
    assert res_esposa["parametros_extraidos"]["cantidad_personas"] == 2
    assert res_esposa["parametros_extraidos"]["en_exterior"] is True

    res_mediodia = modelo_foto.analizar_mensaje("Quiero agendar mañana al mediodía")
    assert res_mediodia["parametros_extraidos"]["fecha_hora"].endswith("12:00")

    res_quince = modelo_foto.analizar_mensaje("Necesito quince fotos de producto")
    assert res_quince["parametros_extraidos"]["cantidad_fotos"] == 15

    res_dia_tarde = modelo_foto.analizar_mensaje("Agendemos el 30 a las 3 de la tarde")
    assert res_dia_tarde["parametros_extraidos"]["fecha_hora"].endswith("-30 15:00")

    res_dia_manana = modelo_foto.analizar_mensaje("El día 30 a las 9 de la mañana")
    assert res_dia_manana["parametros_extraidos"]["fecha_hora"].endswith("-30 09:00")

    res_insulto = modelo_foto.analizar_mensaje("tu negocio es una mierda")
    assert res_insulto["intencion"] == "reclamo_critico"
    assert res_insulto["escalamiento_humano"] is True

    res_fuera = modelo_foto.analizar_mensaje("Hola, ustedes arreglan celulares o hacen plomería?")
    assert res_fuera["parametros_extraidos"].get("fuera_de_alcance") is True
