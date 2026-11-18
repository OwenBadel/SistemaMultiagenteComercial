"""
Generacion e inicializacion automatica de datasets sinteticos realistas
para el entrenamiento de los modelos ML del SistemaMultiagenteComercial.
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
        "nombre": "AutomatizaciÃ³n Inteligente con Flujos IA & Bots",
        "tarifa_hora_cop": 125_000.0,
        "descripcion": "IntegraciÃ³n de asistentes conversacionales, extracciÃ³n de datos y APIs.",
    },
    "ecommerce": {
        "id": 3,
        "nombre": "Tienda Virtual E-Commerce con Pasarela de Pagos",
        "tarifa_hora_cop": 95_000.0,
        "descripcion": "CatÃ¡logo interactivo, carrito, inventario y pagos en lÃ­nea.",
    },
    "app_movil": {
        "id": 4,
        "nombre": "AplicaciÃ³n MÃ³vil Multiplataforma (Android / iOS)",
        "tarifa_hora_cop": 115_000.0,
        "descripcion": "App hÃ­brida conectada a backend REST con notificaciones push.",
    },
    "consultoria_cloud": {
        "id": 5,
        "nombre": "Arquitectura Cloud, MigraciÃ³n y DevOps",
        "tarifa_hora_cop": 110_000.0,
        "descripcion": "Contenedores Docker, CI/CD, seguridad y monitoreo en nube.",
    },
}

CORPUS_NLP_VENTAS: List[Tuple[str, str, str]] = [
    # (texto_mensaje, intencion, tono_cliente)
    ("Hola, buenos dÃ­as, quiero informaciÃ³n sobre sus servicios de desarrollo", "consulta_servicio", "interesado"),
    ("Me gustarÃ­a conocer el portafolio de soluciones que ofrecen", "consulta_servicio", "interesado"),
    ("Â¿QuÃ© servicios manejan para automatizar mi empresa?", "consulta_servicio", "interesado"),
    ("Buenas tardes, estoy buscando apoyo tÃ©cnico para un proyecto digital", "consulta_servicio", "interesado"),
    ("Â¿Me puedes mostrar el catÃ¡logo visual de servicios y precios base?", "consulta_servicio", "interesado"),
    ("Hola, vi su anuncio en Instagram y quiero saber quÃ© ofrecen", "consulta_servicio", "interesado"),
    ("Necesito saber quÃ© planes de desarrollo web o mÃ³vil tienen disponibles", "consulta_servicio", "interesado"),
    ("Me interesa contratar un sistema, Â¿cuÃ©ntame quÃ© servicios manejas?", "consulta_servicio", "interesado"),
    ("Quisiera informaciÃ³n general sobre automatizaciÃ³n e inteligencia artificial", "consulta_servicio", "interesado"),
    ("Hola! Necesito cotizar un servicio de desarrollo_web complejidad 3 con 40 horas", "solicitud_cotizacion", "interesado"),
    ("Quiero una cotizaciÃ³n formal para un ecommerce de 60 horas y urgencia 15 dÃ­as", "solicitud_cotizacion", "interesado"),
    ("Â¿CuÃ¡nto cuesta implementar una automatizacion_ia de 50 horas complejidad 4?", "solicitud_cotizacion", "interesado"),
    ("Por favor cotÃ­zame una app_movil con 80 horas estimadas y mantenimiento", "solicitud_cotizacion", "interesado"),
    ("Necesito presupuesto para consultoria_cloud de 30 horas urgencia 10 dÃ­as", "solicitud_cotizacion", "interesado"),
    ("EnvÃ­ame el valor de un desarrollo web con 45 horas y soporte incluido", "solicitud_cotizacion", "interesado"),
    ("Quiero saber cuÃ¡nto me sale el proyecto web con estas especificaciones", "solicitud_cotizacion", "interesado"),
    ("Cotizar servicio ecommerce para 55 horas con complejidad media", "solicitud_cotizacion", "interesado"),
    ("Necesito una cotizaciÃ³n pero aÃºn no sÃ© cuÃ¡ntas horas ni quÃ© tipo exacto", "solicitud_cotizacion", "indeciso"),
    ("Mmm la verdad el precio me parece algo costoso, no estoy seguro si pueda pagarlo", "objecion_precio", "indeciso"),
    ("EstÃ¡ un poco caro para mi presupuesto actual, lo voy a pensar", "objecion_precio", "indeciso"),
    ("Â¿No hay algÃºn descuento o promociÃ³n? Se me sale del presupuesto", "objecion_precio", "indeciso"),
    ("Siento que el valor es elevado, tengo dudas de cerrar hoy", "objecion_precio", "indeciso"),
    ("Me gusta la propuesta pero el costo estÃ¡ muy alto, Â¿se puede ajustar?", "objecion_precio", "indeciso"),
    ("La verdad estoy indeciso por el precio, necesito una mejor oferta", "objecion_precio", "indeciso"),
    ("Â¿Manejan alguna rebaja o incentivo comercial? El monto supera lo que esperaba", "objecion_precio", "indeciso"),
    ("EstÃ¡ por encima de mi presupuesto, dÃ©jame revisar otras opciones primero", "objecion_precio", "indeciso"),
    ("Tengo mucha duda con esa tarifa, Â¿tienen promociÃ³n vigente?", "objecion_precio", "indeciso"),
    ("Perfecto, acepto la cotizaciÃ³n, quiero agendar una cita para el lunes 10:00", "solicitud_agendamiento", "interesado"),
    ("SÃ­ me sirve la oferta, reservemos una reuniÃ³n el 2026-11-18 10:00", "solicitud_agendamiento", "interesado"),
    ("Quiero confirmar el servicio y agendar fecha de inicio para maÃ±ana a las 15:00", "solicitud_agendamiento", "interesado"),
    ("Acepto el presupuesto con descuento, Â¿tienen disponibilidad el martes a las 9?", "solicitud_agendamiento", "interesado"),
    ("Listo, cerramos trato, agenda mi sesiÃ³n por favor", "solicitud_agendamiento", "interesado"),
    ("Quisiera apartar un espacio en el calendario para arrancar esta semana", "solicitud_agendamiento", "interesado"),
    ("Aprobado, procedamos con la reserva del horario en su agenda", "solicitud_agendamiento", "interesado"),
    ("SÃ­ acepto la propuesta, agÃ©ndame para el horario disponible", "solicitud_agendamiento", "interesado"),
    ("Confirmado el precio, quiero programar la cita de levantamiento", "solicitud_agendamiento", "interesado"),
    ("Esto es un robo, pÃ©simo servicio y nadie me responde lo que necesito", "reclamo_critico", "hostil"),
    ("QuÃ© falta de respeto, voy a denunciarlos y exigir devoluciÃ³n inmediata", "reclamo_critico", "hostil"),
    ("Estoy muy molesto, su sistema no sirve para nada, quiero hablar con el dueÃ±o ya", "reclamo_critico", "hostil"),
    ("Inaceptable la demora y ese cobro abusivo, pÃ¡senme con un humano ahora mismo", "reclamo_critico", "hostil"),
    ("Son unos estafadores, cancelen todo y comunÃ­quenme con el encargado", "reclamo_critico", "hostil"),
    ("Queja formal: incumplieron los acuerdos y el soporte es horrible", "reclamo_critico", "hostil"),
    ("No quiero hablar con ningÃºn bot inÃºtil, exijo atenciÃ³n humana urgente", "reclamo_critico", "hostil"),
    ("QuÃ© porquerÃ­a de atenciÃ³n, me tienen harto con respuestas automÃ¡ticas", "reclamo_critico", "hostil"),
    ("Estoy furioso por el error en mi factura, solucionen esto o cancelo", "reclamo_critico", "hostil"),
]


def generar_tarifas_historicas(n_registros: int = 45, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame realista con histÃ³rico de cotizaciones de servicios."""
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
    Genera un DataFrame realista con histÃ³rico de ofertas y descuentos aplicados.
    Incluye casos normales (descuento 5%-13%) y casos extremos (17%-23%) para que
    KNeighborsRegressor capture ambos regÃ­menes segÃºn indecisiÃ³n y fidelidad.
    """
    rng = np.random.default_rng(seed + 7)
    registros: List[Dict[str, object]] = []

    for idx in range(1, n_registros + 1):
        monto = float(round(rng.uniform(1_800_000.0, 14_500_000.0), -3))
        complejidad = int(rng.integers(1, 6))

        if idx <= 12:
            # IndecisiÃ³n baja: descuentos conservadores (3% - 6%)
            nivel_indecision = round(float(rng.uniform(0.15, 0.45)), 2)
            fidelidad = int(rng.integers(1, 3))
            descuento = round(3.0 + nivel_indecision * 6.5 + (fidelidad - 1) * 1.0, 2)
            cerrado = 1
        elif idx <= 34:
            # IndecisiÃ³n media-alta: zona Ã³ptima (~8% - 12.5%, promedio ~10%)
            nivel_indecision = round(float(rng.uniform(0.58, 0.84)), 2)
            fidelidad = int(rng.integers(1, 3))
            descuento = round(4.8 + nivel_indecision * 7.2 + (fidelidad - 1) * 1.1, 2)
            cerrado = 1
        else:
            # IndecisiÃ³n extrema / campaÃ±as histÃ³ricas agresivas (17.5% - 23%)
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

