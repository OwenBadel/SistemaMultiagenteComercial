"""
Generacion e inicializacion automatica de datasets sinteticos realistas
para el entrenamiento de los modelos ML del SistemaMultiagenteComercial.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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

CORPUS_NLP_VENTAS: List[Tuple[str, str, str]] = [
    # (texto_mensaje, intencion, tono_cliente)
    ("Hola, buenos días, quiero información sobre sus servicios de desarrollo", "consulta_servicio", "interesado"),
    ("Me gustaría conocer el portafolio de soluciones que ofrecen", "consulta_servicio", "interesado"),
    ("¿Qué servicios manejan para automatizar mi empresa?", "consulta_servicio", "interesado"),
    ("Buenas tardes, estoy buscando apoyo técnico para un proyecto digital", "consulta_servicio", "interesado"),
    ("¿Me puedes mostrar el catálogo visual de servicios y precios base?", "consulta_servicio", "interesado"),
    ("Hola, vi su anuncio en Instagram y quiero saber qué ofrecen", "consulta_servicio", "interesado"),
    ("Necesito saber qué planes de desarrollo web o móvil tienen disponibles", "consulta_servicio", "interesado"),
    ("Me interesa contratar un sistema, ¿cuéntame qué servicios manejas?", "consulta_servicio", "interesado"),
    ("Quisiera información general sobre automatización e inteligencia artificial", "consulta_servicio", "interesado"),
    ("Hola! Necesito cotizar un servicio de desarrollo_web complejidad 3 con 40 horas", "solicitud_cotizacion", "interesado"),
    ("Quiero una cotización formal para un ecommerce de 60 horas y urgencia 15 días", "solicitud_cotizacion", "interesado"),
    ("¿Cuánto cuesta implementar una automatizacion_ia de 50 horas complejidad 4?", "solicitud_cotizacion", "interesado"),
    ("Por favor cotízame una app_movil con 80 horas estimadas y mantenimiento", "solicitud_cotizacion", "interesado"),
    ("Necesito presupuesto para consultoria_cloud de 30 horas urgencia 10 días", "solicitud_cotizacion", "interesado"),
    ("Envíame el valor de un desarrollo web con 45 horas y soporte incluido", "solicitud_cotizacion", "interesado"),
    ("Quiero saber cuánto me sale el proyecto web con estas especificaciones", "solicitud_cotizacion", "interesado"),
    ("Cotizar servicio ecommerce para 55 horas con complejidad media", "solicitud_cotizacion", "interesado"),
    ("Necesito una cotización pero aún no sé cuántas horas ni qué tipo exacto", "solicitud_cotizacion", "indeciso"),
    ("Mmm la verdad el precio me parece algo costoso, no estoy seguro si pueda pagarlo", "objecion_precio", "indeciso"),
    ("Está un poco caro para mi presupuesto actual, lo voy a pensar", "objecion_precio", "indeciso"),
    ("¿No hay algún descuento o promoción? Se me sale del presupuesto", "objecion_precio", "indeciso"),
    ("Siento que el valor es elevado, tengo dudas de cerrar hoy", "objecion_precio", "indeciso"),
    ("Me gusta la propuesta pero el costo está muy alto, ¿se puede ajustar?", "objecion_precio", "indeciso"),
    ("La verdad estoy indeciso por el precio, necesito una mejor oferta", "objecion_precio", "indeciso"),
    ("¿Manejan alguna rebaja o incentivo comercial? El monto supera lo que esperaba", "objecion_precio", "indeciso"),
    ("Está por encima de mi presupuesto, déjame revisar otras opciones primero", "objecion_precio", "indeciso"),
    ("Tengo mucha duda con esa tarifa, ¿tienen promoción vigente?", "objecion_precio", "indeciso"),
    ("Perfecto, acepto la cotización, quiero agendar una cita para el lunes 10:00", "solicitud_agendamiento", "interesado"),
    ("Sí me sirve la oferta, reservemos una reunión el 2026-11-18 10:00", "solicitud_agendamiento", "interesado"),
    ("Quiero confirmar el servicio y agendar fecha de inicio para mañana a las 15:00", "solicitud_agendamiento", "interesado"),
    ("Acepto el presupuesto con descuento, ¿tienen disponibilidad el martes a las 9?", "solicitud_agendamiento", "interesado"),
    ("Listo, cerramos trato, agenda mi sesión por favor", "solicitud_agendamiento", "interesado"),
    ("Quisiera apartar un espacio en el calendario para arrancar esta semana", "solicitud_agendamiento", "interesado"),
    ("Aprobado, procedamos con la reserva del horario en su agenda", "solicitud_agendamiento", "interesado"),
    ("Sí acepto la propuesta, agéndame para el horario disponible", "solicitud_agendamiento", "interesado"),
    ("Confirmado el precio, quiero programar la cita de levantamiento", "solicitud_agendamiento", "interesado"),
    ("Esto es un robo, pésimo servicio y nadie me responde lo que necesito", "reclamo_critico", "hostil"),
    ("Qué falta de respeto, voy a denunciarlos y exigir devolución inmediata", "reclamo_critico", "hostil"),
    ("Estoy muy molesto, su sistema no sirve para nada, quiero hablar con el dueño ya", "reclamo_critico", "hostil"),
    ("Inaceptable la demora y ese cobro abusivo, pásenme con un humano ahora mismo", "reclamo_critico", "hostil"),
    ("Son unos estafadores, cancelen todo y comuníquenme con el encargado", "reclamo_critico", "hostil"),
    ("Queja formal: incumplieron los acuerdos y el soporte es horrible", "reclamo_critico", "hostil"),
    ("No quiero hablar con ningún bot inútil, exijo atención humana urgente", "reclamo_critico", "hostil"),
    ("Qué porquería de atención, me tienen harto con respuestas automáticas", "reclamo_critico", "hostil"),
    ("Estoy furioso por el error en mi factura, solucionen esto o cancelo", "reclamo_critico", "hostil"),
]


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

