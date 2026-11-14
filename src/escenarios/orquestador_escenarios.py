"""
Orquestador de escenarios demostrables de principio a fin del SistemaMultiagenteComercial:
- Caso 1A: Cotización exitosa (REQUEST -> AGREE -> RandomForest -> INFORM)
- Caso 1B: Cotización con parámetros incompletos (REQUEST -> AGREE -> FAILURE)
- Caso 2A: Promoción aprobada con KNN (REQUEST -> KNN -> PROPOSE -> ACCEPT-PROPOSAL -> INFORM)
- Caso 2B: Promoción rechazada por margen (REQUEST -> PROPOSE -> REJECT-PROPOSAL -> REFUSE)
- Caso 3A/3B: Agendamiento (QUERY-IF -> AGREE -> CONFIRM / PROPOSE horarios alternativos)
- Caso 4: Escalamiento humano ante tono hostil (INFORM -> CANCEL -> CONFIRM -> pausarYAlertarHumano)

Autoría: Owen Badel Hooker
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agentes.agente_agendamientos import AgenteAgendamientos
from src.agentes.agente_coordinador import AgenteCoordinador
from src.agentes.agente_cotizaciones import AgenteCotizaciones
from src.agentes.agente_promociones import AgentePromociones
from src.agentes.agente_ventas import AgenteVentas
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.comunicacion.spade_runtime import detener_agentes_spade, iniciar_agentes_spade
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.ml.datasets import inicializar_datasets
from src.ml.modelos_ml import (
    ModeloCotizacionesRF,
    ModeloNLPVentas,
    ModeloPromocionesKNN,
)
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


class SistemaComercialOrquestador:
    """
    Ensambla el Entorno, la Organización MAS-ML, los modelos Scikit-Learn y los
    5 agentes SPADE, exponiendo métodos asíncronos reproducibles para cada caso.
    """

    def __init__(
        self,
        ruta_data: Optional[Path] = None,
        usar_xmpp_embebido: bool = True,
        consola: Optional[Console] = None,
        silencioso: bool = False,
    ) -> None:
        raiz = Path(__file__).resolve().parents[2]
        self.ruta_data = ruta_data or (raiz / "data")
        self.usar_xmpp_embebido = usar_xmpp_embebido
        self.consola = consola or Console()
        self.silencioso = silencioso

        # 1. Inicialización automática de datasets sintéticos (>= 30 registros)
        self.df_tarifas, self.df_ofertas = inicializar_datasets(self.ruta_data)

        # 2. Entrenamiento inicial de modelos Scikit-Learn
        self.modelo_nlp = ModeloNLPVentas()
        self.modelo_rf = ModeloCotizacionesRF(self.df_tarifas)
        self.modelo_knn = ModeloPromocionesKNN(self.df_ofertas)

        # 3. Monitor FIPA-ACL y Entorno MAS-ML
        self.monitor = MonitorFIPAACL(consola=self.consola, silencioso=self.silencioso)
        self.entorno = EntornoVirtualAtencion(
            tarifas_historicas=self.df_tarifas,
            historico_ofertas=self.df_ofertas,
            logger_evento=self.monitor.registrar_evento_entorno,
        )

        # 4. Organización MAS-ML
        self.organizacion = SistemaMultiagenteComercial(entorno=self.entorno)

        # 5. Instanciación de los 5 Agentes SPADE
        self.coordinador = AgenteCoordinador(
            entorno=self.entorno, organizacion=self.organizacion, monitor=self.monitor
        )
        self.ventas = AgenteVentas(
            entorno=self.entorno,
            organizacion=self.organizacion,
            monitor=self.monitor,
            modelo_nlp=self.modelo_nlp,
        )
        self.cotizaciones = AgenteCotizaciones(
            entorno=self.entorno,
            organizacion=self.organizacion,
            monitor=self.monitor,
            modelo_rf=self.modelo_rf,
        )
        self.promociones = AgentePromociones(
            entorno=self.entorno,
            organizacion=self.organizacion,
            monitor=self.monitor,
            modelo_knn=self.modelo_knn,
        )
        self.agendamientos = AgenteAgendamientos(
            entorno=self.entorno, organizacion=self.organizacion, monitor=self.monitor
        )
        self.agentes = [
            self.coordinador,
            self.ventas,
            self.cotizaciones,
            self.promociones,
            self.agendamientos,
        ]
        self._iniciado = False

    async def iniciar(self) -> None:
        """Arranca los 5 agentes en el runtime SPADE."""
        if not self._iniciado:
            await iniciar_agentes_spade(
                self.agentes, usar_xmpp_embebido=self.usar_xmpp_embebido
            )
            self._iniciado = True

    async def detener(self) -> None:
        """Detiene todos los agentes SPADE."""
        if self._iniciado:
            await detener_agentes_spade(
                self.agentes, usar_xmpp_embebido=self.usar_xmpp_embebido
            )
            self._iniciado = False

    def imprimir_resumen_inicial(self) -> None:
        """Muestra la arquitectura MAS-ML y métricas de entrenamiento ML."""
        if self.silencioso:
            return
        tabla = Table(
            title="Resumen Arquitectónico MAS-ML & Modelos Scikit-Learn",
            show_header=True,
            header_style="bold cyan",
        )
        tabla.add_column("Agente SPADE (<<agent>>)", style="bold white")
        tabla.add_column("Rol (<<role>>)", style="green")
        tabla.add_column("Arquitectura / Motor ML", style="yellow")
        tabla.add_column("Estado / Métrica", style="bright_blue")

        tabla.add_row(
            "AgenteCoordinador",
            "Coordinador",
            "Basado en utilidad (<<control>> x4)",
            "Activo · Orquestador",
        )
        tabla.add_row(
            "AgenteVentas",
            "EjecutivoVentas",
            "NLP (TF-IDF + LogReg) + Utilidad",
            "Activo · Puerta única",
        )
        tabla.add_row(
            "AgenteCotizaciones",
            "LiquidadorCotizaciones",
            "RandomForestRegressor + Metas",
            f"R²={self.modelo_rf.r2_entrenamiento:.4f} ({len(self.df_tarifas)} reg.)",
        )
        tabla.add_row(
            "AgentePromociones",
            "AnalistaPromociones",
            "KNeighborsRegressor (KNN) + Utilidad",
            f"R²={self.modelo_knn.r2_entrenamiento:.4f} ({len(self.df_ofertas)} reg.)",
        )
        tabla.add_row(
            "AgenteAgendamientos",
            "GestorAgenda",
            "Basado en metas (Calendario)",
            f"{len(self.entorno.agendaVirtual.bloques)} bloques cargados",
        )
        self.consola.print(tabla)

    async def _despachar_y_esperar(
        self,
        tipo_evento: str,
        payload: Dict[str, Any],
        conversation_id: str,
        esperar_coordinador: bool = False,
        timeout: float = 8.0,
    ) -> Dict[str, Any]:
        """Dispara un eventoLlegadaMensaje y espera el término del ciclo FIPA."""
        self.ventas.reiniciar_evento_ciclo()
        self.coordinador.reiniciar_evento_ciclo()

        evento = self.entorno.eventoLlegadaMensaje(tipo_evento=tipo_evento, payload=payload)
        self.ventas.procesar_evento_desde_entorno(
            evento=evento, conversation_id=conversation_id
        )

        await asyncio.wait_for(self.ventas.evento_ciclo_completado.wait(), timeout=timeout)
        if esperar_coordinador:
            await asyncio.wait_for(
                self.coordinador.evento_ciclo_completado.wait(), timeout=timeout
            )
        await asyncio.sleep(0.05)
        return dict(self.ventas.ultimo_resultado)

    def _titulo_caso(self, titulo: str, subtitulo: str) -> None:
        if not self.silencioso:
            self.consola.print(
                Panel(
                    f"[bold white]{subtitulo}[/bold white]",
                    title=f"[bold bright_cyan]{titulo}[/bold bright_cyan]",
                    border_style="bright_cyan",
                )
            )

    # ------------------------------------------------------------------
    # ESCENARIOS DEMOSTRATIVOS EXIGIDOS
    # ------------------------------------------------------------------
    async def ejecutar_caso_1a(self) -> Dict[str, Any]:
        """Caso 1A: Flujo de Cotización exitosa (REQUEST -> AGREE -> RandomForest -> INFORM)."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 1A — ESCENARIO 1 (Rama alt: Cálculo Exitoso)",
            "REQUEST -> AGREE -> consulta tarifasYMargenes -> RandomForestRegressor -> INFORM",
        )
        # Paso 1: Consulta inicial de servicio -> enviarCatalogoVisual
        res_catalogo = await self._despachar_y_esperar(
            tipo_evento="consulta_servicio",
            payload={
                "id_cliente": "cli_101",
                "nombre": "TechCaribe SAS",
                "canal": "WhatsApp",
                "texto": "Hola, quiero conocer sus servicios de desarrollo y automatización",
            },
            conversation_id="conv-1a-catalogo",
        )
        # Paso 2: Cliente envía parámetros completos -> escalamiento -> RandomForest -> INFORM
        res_cotizacion = await self._despachar_y_esperar(
            tipo_evento="parametros_servicio",
            payload={
                "id_cliente": "cli_101",
                "nombre": "TechCaribe SAS",
                "canal": "WhatsApp",
                "texto": "Necesito cotizar un ecommerce complejidad 3 con 45 horas y urgencia 12 dias",
                "parametros_servicio": {
                    "tipo_servicio": "ecommerce",
                    "complejidad": 3,
                    "horas_estimadas": 45,
                    "urgencia_dias": 12,
                    "incluye_mantenimiento": 1,
                },
            },
            conversation_id="conv-1a-cotiz",
        )
        return {
            "caso": "1A",
            "catalogo": res_catalogo,
            "cotizacion": res_cotizacion,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_caso_1b(self) -> Dict[str, Any]:
        """Caso 1B: Flujo de Cotización con parámetros incompletos (REQUEST -> AGREE -> FAILURE)."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 1B — ESCENARIO 1 (Rama else: Parámetros Incompletos)",
            "REQUEST -> AGREE -> validación Liquidador -> FAILURE -> REQUEST aclaración",
        )
        res = await self._despachar_y_esperar(
            tipo_evento="parametros_servicio",
            payload={
                "id_cliente": "cli_102",
                "nombre": "Comercializadora del Norte",
                "canal": "Instagram",
                "texto": "Quiero cotizar un desarrollo web pero aún no defino horas ni complejidad",
                "parametros_servicio": {
                    "tipo_servicio": "desarrollo_web",
                    # Faltan deliberadamente horas_estimadas y complejidad
                },
            },
            conversation_id="conv-1b-incompleto",
        )
        return {
            "caso": "1B",
            "resultado": res,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_caso_2a(self) -> Dict[str, Any]:
        """Caso 2A: Promoción aprobada con KNN (REQUEST -> KNN -> PROPOSE -> ACCEPT-PROPOSAL -> INFORM)."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 2A — ESCENARIO 2 (Rama alt: Promoción Aprobada con Regresión KNN)",
            "Objeción precio (indeciso) -> KNN (~10%) -> PROPOSE -> ACCEPT-PROPOSAL -> INFORM",
        )
        res = await self._despachar_y_esperar(
            tipo_evento="objecion_precio",
            payload={
                "id_cliente": "cli_201",
                "nombre": "Inversiones Bolívar",
                "canal": "WhatsApp",
                "fidelidad": 1,
                "complejidad": 3,
                "cotizacion_actual": 5_200_000.0,
                "nivel_indecision": 0.72,
                "texto": "El precio me parece un poco costoso para mi presupuesto actual, tengo dudas",
            },
            conversation_id="conv-2a-promo-ok",
        )
        return {
            "caso": "2A",
            "resultado": res,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_caso_2b(self) -> Dict[str, Any]:
        """Caso 2B: Promoción rechazada por margen (REQUEST -> PROPOSE -> REJECT-PROPOSAL -> REFUSE)."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 2B — ESCENARIO 2 (Rama else: Promoción Rechazada por Margen)",
            "Resistencia extrema -> KNN (~20%) -> PROPOSE -> REJECT-PROPOSAL -> REFUSE",
        )
        res = await self._despachar_y_esperar(
            tipo_evento="objecion_precio",
            payload={
                "id_cliente": "cli_202",
                "nombre": "Grupo Atlántico",
                "canal": "Facebook",
                "fidelidad": 3,
                "complejidad": 4,
                "cotizacion_actual": 8_000_000.0,
                "nivel_indecision": 0.96,
                "texto": "Me parece demasiado alto ese precio, necesito una rebaja muy grande o no cierro",
            },
            conversation_id="conv-2b-promo-rej",
        )
        return {
            "caso": "2B",
            "resultado": res,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_caso_3a(self) -> Dict[str, Any]:
        """Caso 3A (Opcional): Agendamiento exitoso con QUERY-IF, AGREE y CONFIRM."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 3A — ESCENARIO 3 (Agendamiento Exitoso: QUERY-IF -> AGREE -> CONFIRM)",
            "Validación de calendario libre y registro definitivo de cita en agendaVirtual",
        )
        # Aseguramos que el bloque esté libre antes de la prueba
        self.entorno.eventoCambioDisponibilidad(
            fecha_hora="2026-11-18 10:00", estado="LIBRE", origen="sistema_demo"
        )
        res = await self._despachar_y_esperar(
            tipo_evento="aceptar_cotizacion",
            payload={
                "id_cliente": "cli_301",
                "nombre": "TechCaribe SAS",
                "canal": "WhatsApp",
                "cotizacion_aprobada": True,
                "fecha_hora": "2026-11-18 10:00",
                "texto": "Perfecto, acepto la cotización, quiero agendar la cita el 2026-11-18 10:00",
            },
            conversation_id="conv-3a-agenda-ok",
        )
        return {
            "caso": "3A",
            "resultado": res,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_caso_3b(self) -> Dict[str, Any]:
        """Caso 3B (Opcional): Superposición horaria tras eventoCambioDisponibilidad()."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 3B — ESCENARIO 3 (Superposición Horaria y Horarios Alternativos)",
            "eventoCambioDisponibilidad() -> QUERY-IF -> INFORM false -> PROPOSE alternativos",
        )
        # Demuestra la característica Dinámica del entorno: el dueño ocupa el bloque
        self.entorno.eventoCambioDisponibilidad(
            fecha_hora="2026-11-18 14:00",
            estado="OCUPADO",
            origen="profesional_humano",
            cliente="Empresa Alfa SAS",
        )
        res = await self._despachar_y_esperar(
            tipo_evento="aceptar_cotizacion",
            payload={
                "id_cliente": "cli_302",
                "nombre": "Logística Express",
                "canal": "WhatsApp",
                "cotizacion_aprobada": True,
                "fecha_hora": "2026-11-18 14:00",
                "texto": "Acepto la oferta, quiero reservar el 2026-11-18 14:00",
            },
            conversation_id="conv-3b-agenda-alt",
        )
        return {
            "caso": "3B",
            "resultado": res,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_caso_4(self) -> Dict[str, Any]:
        """Caso 4 (Opcional): Cancelación automática y escalamiento humano ante tono hostil."""
        self.monitor.limpiar()
        self._titulo_caso(
            "CASO 4 — ESCENARIO 4 (Escalamiento Humano: INFORM -> CANCEL -> CONFIRM)",
            "Detección de tono hostil por NLP -> suspensión de automatización -> alerta humana",
        )
        res_ventas = await self._despachar_y_esperar(
            tipo_evento="reclamo_critico_cliente",
            payload={
                "id_cliente": "cli_401",
                "nombre": "Cliente Inconforme",
                "canal": "WhatsApp",
                "texto": "Esto es un robo, pésimo servicio, no quiero hablar con bots, pásenme con el dueño ya",
            },
            conversation_id="conv-4-hostil",
            esperar_coordinador=True,
        )
        return {
            "caso": "4",
            "resultado_ventas": res_ventas,
            "alerta_humana": dict(self.coordinador.ultimo_resultado),
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }

    async def ejecutar_mensaje_personalizado(
        self,
        texto_cliente: str,
        canal: str = "WhatsApp",
        cotizacion_actual: float = 4_800_000.0,
    ) -> Dict[str, Any]:
        """Permite al profesor o usuario enviar un mensaje libre en tiempo real."""
        self.monitor.limpiar()
        self._titulo_caso(
            "SIMULACIÓN EN VIVO — MENSAJE PERSONALIZADO",
            f"Canal: {canal} | Entrada: '{texto_cliente}'",
        )
        analisis_previo = self.modelo_nlp.analizar_mensaje(texto_cliente)
        esperar_coord = analisis_previo["tono_cliente"] == "hostil"
        res = await self._despachar_y_esperar(
            tipo_evento="mensaje_libre",
            payload={
                "id_cliente": "cli_live",
                "nombre": "Visitante En Vivo",
                "canal": canal,
                "texto": texto_cliente,
                "cotizacion_actual": cotizacion_actual,
                "cotizacion_aprobada": True,
                "parametros_servicio": analisis_previo.get("parametros_extraidos", {}),
            },
            conversation_id="conv-live",
            esperar_coordinador=esperar_coord,
        )
        return {
            "caso": "LIVE",
            "nlp": analisis_previo,
            "resultado": res,
            "mensajes_fipa": list(self.monitor.historial_mensajes),
        }
