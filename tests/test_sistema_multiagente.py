"""
Suite de pruebas unitarias e integración E2E para SistemaMultiagenteComercial.
Valida:
- Metamodelo MAS-ML (Organización, Entorno, 5 Roles, 4 Control, 5 Dependencias).
- Datasets sintéticos (>=30 registros) y modelos Scikit-Learn (RF, KNN, NLP).
- Mensajería FIPA-ACL sobre SPADE y ejecución completa de Casos 1A, 1B, 2A, 2B, 3A, 3B y 4.

Autoría: Owen Badel Hooker
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import pytest

from src.comunicacion.fipa_acl import ONTOLOGIA_COMERCIAL, PROTOCOLOS_VALIDOS
from src.escenarios.orquestador_escenarios import SistemaComercialOrquestador


def _extraer_secuencia_performativas(mensajes_fipa: List[dict]) -> List[str]:
    return [m["fipa_json"][":performative"] for m in mensajes_fipa]


def _extraer_pares_emisor_receptor(mensajes_fipa: List[dict]) -> List[str]:
    return [m["etiqueta_diapositiva"].splitlines()[0] for m in mensajes_fipa]


def test_estructura_masml_organizacion_roles_y_dependencias(tmp_path: Path) -> None:
    orq = SistemaComercialOrquestador(
        ruta_data=tmp_path / "data",
        usar_xmpp_embebido=False,
        silencioso=True,
    )

    # 1. Organización posee por composición (<<owns>>) los 5 roles
    assert set(orq.organizacion.roles.keys()) == {
        "Coordinador",
        "EjecutivoVentas",
        "LiquidadorCotizaciones",
        "AnalistaPromociones",
        "GestorAgenda",
    }
    assert "SistemaMultiagenteComercial" in orq.entorno.habitantes

    # 2. Atributos del entorno <<environment>> EntornoVirtualAtencion
    assert hasattr(orq.entorno, "plataformas")
    assert hasattr(orq.entorno, "perfilesClientes")
    assert hasattr(orq.entorno, "tarifasYMargenes")
    assert hasattr(orq.entorno, "agendaVirtual")

    # 3. Relaciones <<control>> del Coordinador
    for subordinado in (
        "EjecutivoVentas",
        "AnalistaPromociones",
        "LiquidadorCotizaciones",
        "GestorAgenda",
    ):
        assert orq.organizacion.verificar_control("Coordinador", subordinado)

    # 4. Las 5 dependencias formales ([Dependiente] <--- [Proveedor])
    esperadas = [
        ("Coordinador", "LiquidadorCotizaciones"),
        ("Coordinador", "AnalistaPromociones"),
        ("EjecutivoVentas", "Coordinador"),
        ("GestorAgenda", "LiquidadorCotizaciones"),
        ("Coordinador", "GestorAgenda"),
    ]
    for dep, prov in esperadas:
        assert orq.organizacion.verificar_dependencia(dep, prov)

    # 5. Reglas organizacionales
    assert not orq.organizacion.aplicar_regla_escalamiento(
        "EjecutivoVentas", "LiquidadorCotizaciones"
    )
    assert not orq.organizacion.aplicar_regla_canal_centralizado_promociones(
        "AnalistaPromociones", "EjecutivoVentas"
    )
    assert orq.organizacion.aplicar_regla_canal_centralizado_promociones(
        "AnalistaPromociones", "Coordinador"
    )


def test_datasets_sinteticos_y_modelos_sklearn(tmp_path: Path) -> None:
    ruta_data = tmp_path / "data"
    orq = SistemaComercialOrquestador(
        ruta_data=ruta_data,
        usar_xmpp_embebido=False,
        silencioso=True,
    )

    assert len(orq.df_tarifas) >= 30
    assert len(orq.df_ofertas) >= 30
    assert (ruta_data / "tarifas_historicas.csv").exists()
    assert (ruta_data / "historico_ofertas.csv").exists()

    # RandomForestRegressor en AgenteCotizaciones
    pred_rf = orq.modelo_rf.predecir_cotizacion(
        {
            "tipo_servicio": "ecommerce",
            "complejidad": 3,
            "horas_estimadas": 50,
            "urgencia_dias": 15,
            "incluye_mantenimiento": 1,
        }
    )
    assert pred_rf["valor_total"] > 1_000_000
    assert orq.modelo_rf.r2_entrenamiento >= 0.85

    # KNeighborsRegressor en AgentePromociones
    pred_knn_viable = orq.modelo_knn.predecir_descuento_optimo(
        monto_cotizado=5_200_000.0, nivel_indecision=0.72, complejidad=3, fidelidad_cliente=1
    )
    assert 5.0 <= pred_knn_viable["descuento_porciento"] <= 15.0

    pred_knn_exceso = orq.modelo_knn.predecir_descuento_optimo(
        monto_cotizado=8_000_000.0, nivel_indecision=0.96, complejidad=4, fidelidad_cliente=3
    )
    assert pred_knn_exceso["descuento_porciento"] > 15.0

    # NLP de AgenteVentas (interesado, indeciso, hostil)
    assert orq.modelo_nlp.analizar_mensaje("Quiero información de servicios")["tono_cliente"] == "interesado"
    assert orq.modelo_nlp.analizar_mensaje("Está muy caro para mi presupuesto")["tono_cliente"] == "indeciso"
    assert orq.modelo_nlp.analizar_mensaje("Esto es un robo, pésimo servicio")["tono_cliente"] == "hostil"


def test_escenarios_completos_fipa_acl_spade(tmp_path: Path) -> None:
    async def correr_todos() -> None:
        orq = SistemaComercialOrquestador(
            ruta_data=tmp_path / "data",
            usar_xmpp_embebido=False,
            silencioso=True,
        )
        await orq.iniciar()
        try:
            # Caso 1A: Cotización exitosa (REQUEST -> AGREE -> REQUEST -> AGREE -> INFORM -> INFORM)
            r1a = await orq.ejecutar_caso_1a()
            assert r1a["catalogo"]["accion"] == "catalogo_enviado"
            assert r1a["cotizacion"]["resultado"] == "cotizacion_entregada"
            assert r1a["cotizacion"]["valor_total"] > 0
            assert _extraer_secuencia_performativas(r1a["mensajes_fipa"]) == [
                "REQUEST",
                "AGREE",
                "REQUEST",
                "AGREE",
                "INFORM",
                "INFORM",
            ]
            # Verificación formato diapositiva 3 líneas + metadata FIPA
            primer_msg = r1a["mensajes_fipa"][0]
            lineas = primer_msg["etiqueta_diapositiva"].splitlines()
            assert len(lineas) == 4
            assert lineas[0] == "[AgenteVentas -> AgenteCoordinador]"
            assert lineas[1] == "REQUEST"
            assert lineas[3] == "<<ACL:REQUEST>>"
            assert primer_msg["fipa_json"][":ontology"] == ONTOLOGIA_COMERCIAL
            assert primer_msg["fipa_json"][":protocol"] in PROTOCOLOS_VALIDOS

            # Caso 1B: Parámetros incompletos (REQUEST -> AGREE -> REQUEST -> AGREE -> FAILURE -> REQUEST)
            r1b = await orq.ejecutar_caso_1b()
            assert r1b["resultado"]["resultado"] == "parametros_incompletos"
            assert "horas_estimadas" in r1b["resultado"]["faltantes"]
            assert _extraer_secuencia_performativas(r1b["mensajes_fipa"]) == [
                "REQUEST",
                "AGREE",
                "REQUEST",
                "AGREE",
                "FAILURE",
                "REQUEST",
            ]

            # Caso 2A: Promoción aprobada con KNN
            r2a = await orq.ejecutar_caso_2a()
            assert r2a["resultado"]["resultado"] == "promocion_presentada"
            assert _extraer_secuencia_performativas(r2a["mensajes_fipa"]) == [
                "REQUEST",
                "REQUEST",
                "PROPOSE",
                "ACCEPT-PROPOSAL",
                "INFORM",
                "INFORM",
            ]

            # Caso 2B: Promoción rechazada por margen
            r2b = await orq.ejecutar_caso_2b()
            assert r2b["resultado"]["resultado"] == "descuento_rechazado"
            assert _extraer_secuencia_performativas(r2b["mensajes_fipa"]) == [
                "REQUEST",
                "REQUEST",
                "PROPOSE",
                "REJECT-PROPOSAL",
                "REFUSE",
            ]

            # Caso 3A: Agendamiento exitoso (QUERY-IF -> INFORM -> REQUEST -> AGREE -> CONFIRM -> INFORM)
            r3a = await orq.ejecutar_caso_3a()
            assert r3a["resultado"]["resultado"] == "cita_confirmada"
            assert _extraer_secuencia_performativas(r3a["mensajes_fipa"]) == [
                "REQUEST",
                "QUERY-IF",
                "INFORM",
                "REQUEST",
                "AGREE",
                "CONFIRM",
                "INFORM",
            ]

            # Caso 3B: Superposición horaria (QUERY-IF -> INFORM false -> PROPOSE -> INFORM)
            r3b = await orq.ejecutar_caso_3b()
            assert r3b["resultado"]["resultado"] == "horarios_alternativos"
            assert len(r3b["resultado"]["alternativas"]) >= 1
            assert _extraer_secuencia_performativas(r3b["mensajes_fipa"]) == [
                "REQUEST",
                "QUERY-IF",
                "INFORM",
                "PROPOSE",
                "INFORM",
            ]

            # Caso 4: Tono hostil y escalamiento humano (INFORM -> CANCEL -> CONFIRM)
            r4 = await orq.ejecutar_caso_4()
            assert r4["resultado_ventas"]["resultado"] == "automatizacion_pausada"
            assert r4["alerta_humana"]["estado"] == "alerta_recibida_por_profesional"
            assert _extraer_secuencia_performativas(r4["mensajes_fipa"]) == [
                "INFORM",
                "CANCEL",
                "CONFIRM",
            ]
        finally:
            await orq.detener()

    asyncio.run(correr_todos())
