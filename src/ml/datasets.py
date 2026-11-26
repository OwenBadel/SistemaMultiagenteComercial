"""
Generacion e inicializacion automatica de datasets sinteticos realistas
para el entrenamiento de los modelos ML numericos del SistemaMultiagenteComercial
(RandomForestRegressor en AgenteCotizaciones y KNeighborsRegressor en AgentePromociones).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

CATALOGO_SERVICIOS: Dict[str, Dict[str, object]] = {
    "desarrollo_web": {
        "id": 1,
        "nombre": "Desarrollo Web Corporativo / Landing & Portal",
        "tarifa_hora_cop": 85_000.0,
        "descripcion": "Sitio web responsivo, panel administrativo y despliegue en la nube.",
    },
    "automatizacion_ia": {
        "id": 2,
        "nombre": "Automatización Inteligente con Flujos IA & Bots",
        "tarifa_hora_cop": 125_000.0,
        "descripcion": "Integración de asistentes conversacionales, extracción de datos y APIs.",
    },
    "ecommerce": {
        "id": 3,
        "nombre": "Tienda Virtual E-Commerce con Pasarela de Pagos",
        "tarifa_hora_cop": 95_000.0,
        "descripcion": "Catálogo interactivo, carrito, inventario y pagos en línea.",
    },
    "app_movil": {
        "id": 4,
        "nombre": "Aplicación Móvil Multiplataforma (Android / iOS)",
        "tarifa_hora_cop": 115_000.0,
        "descripcion": "App híbrida conectada a backend REST con notificaciones push.",
    },
    "consultoria_cloud": {
        "id": 5,
        "nombre": "Arquitectura Cloud, Migración y DevOps",
        "tarifa_hora_cop": 110_000.0,
        "descripcion": "Contenedores Docker, CI/CD, seguridad y monitoreo en nube.",
    },
}


def generar_tarifas_historicas(n_registros: int = 45, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame realista con histórico de cotizaciones de servicios."""
    rng = np.random.default_rng(seed)
    nombres_servicios = list(CATALOGO_SERVICIOS.keys())
    registros: List[Dict[str, object]] = []

    for idx in range(1, n_registros + 1):
        servicio = nombres_servicios[(idx - 1) % len(nombres_servicios)]
        meta = CATALOGO_SERVICIOS[servicio]
        servicio_id = int(meta["id"])
        tarifa_hora = float(meta["tarifa_hora_cop"])

        complejidad = int(rng.integers(1, 6))
        horas_estimadas = int(rng.integers(12, 141))
        urgencia_dias = int(rng.integers(4, 46))
        incluye_mantenimiento = int(rng.integers(0, 2))

        factor_complejidad = 1.0 + 0.14 * (complejidad - 1)
        factor_urgencia = 1.15 if urgencia_dias <= 7 else (1.06 if urgencia_dias <= 15 else 1.0)
        adicional_soporte = 380_000.0 if incluye_mantenimiento == 1 else 0.0

        costo_base = horas_estimadas * tarifa_hora * factor_complejidad
        ruido = float(rng.normal(1.0, 0.02))
        costo_total = round((costo_base * factor_urgencia + adicional_soporte) * ruido, -3)
        margen_estimado = round(float(rng.uniform(0.32, 0.40)), 3)

        registros.append(
            {
                "id_cotizacion": f"COT-{idx:03d}",
                "tipo_servicio": servicio,
                "tipo_servicio_id": servicio_id,
                "complejidad": complejidad,
                "horas_estimadas": horas_estimadas,
                "urgencia_dias": urgencia_dias,
                "incluye_mantenimiento": incluye_mantenimiento,
                "costo_base_cop": round(costo_base, 2),
                "costo_total_cop": float(costo_total),
                "margen_estimado": margen_estimado,
            }
        )

    return pd.DataFrame(registros)


def generar_historico_ofertas(n_registros: int = 45, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame realista con histórico de ofertas y descuentos aplicados.
    Incluye casos normales (descuento 5%-13%) y casos extremos (17%-23%) para que
    KNeighborsRegressor capture ambos regímenes según indecisión y fidelidad.
    """
    rng = np.random.default_rng(seed + 7)
    registros: List[Dict[str, object]] = []

    for idx in range(1, n_registros + 1):
        monto = float(round(rng.uniform(1_800_000.0, 14_500_000.0), -3))
        complejidad = int(rng.integers(1, 6))

        if idx <= 12:
            # Indecisión baja: descuentos conservadores (3% - 6%)
            nivel_indecision = round(float(rng.uniform(0.15, 0.45)), 2)
            fidelidad = int(rng.integers(1, 3))
            descuento = round(3.0 + nivel_indecision * 6.5 + (fidelidad - 1) * 1.0, 2)
            cerrado = 1
        elif idx <= 34:
            # Indecisión media-alta: zona óptima (~8% - 12.5%, promedio ~10%)
            nivel_indecision = round(float(rng.uniform(0.58, 0.84)), 2)
            fidelidad = int(rng.integers(1, 3))
            descuento = round(4.8 + nivel_indecision * 7.2 + (fidelidad - 1) * 1.1, 2)
            cerrado = 1
        else:
            # Indecisión extrema / campañas históricas agresivas (17.5% - 23%)
            nivel_indecision = round(float(rng.uniform(0.88, 0.99)), 2)
            fidelidad = 3
            descuento = round(16.5 + (nivel_indecision - 0.85) * 42.0, 2)
            cerrado = int(rng.choice([0, 1], p=[0.4, 0.6]))

        registros.append(
            {
                "id_oferta": f"OFE-{idx:03d}",
                "monto_cotizado": monto,
                "nivel_indecision": nivel_indecision,
                "complejidad": complejidad,
                "fidelidad_cliente": fidelidad,
                "porcentaje_descuento_optimo": float(round(descuento, 2)),
                "cerrado_exitosamente": cerrado,
            }
        )

    return pd.DataFrame(registros)
