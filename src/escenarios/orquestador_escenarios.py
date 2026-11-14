"""
Servicio principal del SistemaMultiagenteComercial.
Ensambla el entorno EntornoVirtualAtencion conectado a Meta Graph API,
la organizacion MAS-ML SistemaMultiagenteComercial, los modelos de Scikit-Learn
y los 5 agentes SPADE sobre XMPP embebido.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from src.agentes.agente_agendamientos import AgenteAgendamientos
from src.agentes.agente_coordinador import AgenteCoordinador
from src.agentes.agente_cotizaciones import AgenteCotizaciones
from src.agentes.agente_promociones import AgentePromociones
from src.agentes.agente_ventas import AgenteVentas
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.comunicacion.spade_runtime import detener_agentes_spade, iniciar_agentes_spade
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.integracion_meta.cliente_meta import ClienteMetaGraphAPI
from src.integracion_meta.configuracion import ConfiguracionMeta
from src.ml.datasets import inicializar_datasets
from src.ml.modelos_ml import (
    ModeloCotizacionesRF,
    ModeloNLPVentas,
    ModeloPromocionesKNN,
)
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


class ServicioMultiagenteComercial:
    """
    Orquestador de ejecucion real para el SistemaMultiagenteComercial
    conectado a canales de Meta (WhatsApp, Instagram, Messenger).
    """

    def __init__(
        self,
        ruta_data: Optional[Path] = None,
        config_meta: Optional[ConfiguracionMeta] = None,
        cliente_meta: Optional[ClienteMetaGraphAPI] = None,
        usar_xmpp_embebido: bool = True,
        consola: Optional[Console] = None,
        silencioso: bool = False,
    ) -> None:
        raiz = Path(__file__).resolve().parents[2]
        self.ruta_data = ruta_data or (raiz / "data")
        self.config_meta = config_meta or ConfiguracionMeta.desde_entorno(raiz / ".env")
        self.cliente_meta = cliente_meta or ClienteMetaGraphAPI(self.config_meta)
        self.usar_xmpp_embebido = usar_xmpp_embebido
        self.consola = consola or Console()
        self.silencioso = silencioso

        # 1. Inicializacion de datasets historicos (>= 30 registros)
        self.df_tarifas, self.df_ofertas = inicializar_datasets(self.ruta_data)

        # 2. Entrenamiento de modelos Scikit-Learn
        self.modelo_nlp = ModeloNLPVentas()
        self.modelo_rf = ModeloCotizacionesRF(self.df_tarifas)
        self.modelo_knn = ModeloPromocionesKNN(self.df_ofertas)

        # 3. Monitor FIPA-ACL y Entorno MAS-ML conectado a Meta
        self.monitor = MonitorFIPAACL(consola=self.consola, silencioso=self.silencioso)
        self.entorno = EntornoVirtualAtencion(
            tarifas_historicas=self.df_tarifas,
            historico_ofertas=self.df_ofertas,
            logger_evento=self.monitor.registrar_evento_entorno,
            cliente_meta=self.cliente_meta,
        )

        # 4. Organizacion MAS-ML
        self.organizacion = SistemaMultiagenteComercial(entorno=self.entorno)

        # 5. Instanciacion de los 5 Agentes SPADE
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
        self._lock_flujo = asyncio.Lock()

    async def iniciar(self) -> None:
        """Arranca los 5 agentes en el runtime SPADE."""
        if not self._iniciado:
            await iniciar_agentes_spade(
                self.agentes, usar_xmpp_embebido=self.usar_xmpp_embebido
            )
            self._iniciado = True

    async def detener(self) -> None:
        """Detiene limpiamente los 5 agentes SPADE."""
        if self._iniciado:
            await detener_agentes_spade(
                self.agentes, usar_xmpp_embebido=self.usar_xmpp_embebido
            )
            self._iniciado = False

    def imprimir_resumen_inicial(self) -> None:
        """Muestra la arquitectura MAS-ML y estado de canales Meta en consola."""
        if self.silencioso:
            return
        tabla = Table(
            title="SistemaMultiagenteComercial - Arquitectura MAS-ML & Modelos ML",
            show_header=True,
            header_style="bold cyan",
        )
        tabla.add_column("Agente SPADE (<<agent>>)", style="bold white")
        tabla.add_column("Rol (<<role>>)", style="green")
        tabla.add_column("Arquitectura / Motor ML", style="yellow")
        tabla.add_column("Estado / Metrica", style="bright_blue")

        tabla.add_row(
            "AgenteCoordinador",
            "Coordinador",
            "Basado en utilidad (<<control>> x4)",
            "Activo | Orquestador",
        )
        tabla.add_row(
            "AgenteVentas",
            "EjecutivoVentas",
            "NLP (TF-IDF + LogReg) + Utilidad",
            "Activo | Puerta unica",
        )
        tabla.add_row(
            "AgenteCotizaciones",
            "LiquidadorCotizaciones",
            "RandomForestRegressor + Metas",
            f"R2={self.modelo_rf.r2_entrenamiento:.4f} ({len(self.df_tarifas)} reg.)",
        )
        tabla.add_row(
            "AgentePromociones",
            "AnalistaPromociones",
            "KNeighborsRegressor (KNN) + Utilidad",
            f"R2={self.modelo_knn.r2_entrenamiento:.4f} ({len(self.df_ofertas)} reg.)",
        )
        tabla.add_row(
            "AgenteAgendamientos",
            "GestorAgenda",
            "Basado en metas (Calendario)",
            f"{len(self.entorno.agendaVirtual.bloques)} bloques cargados",
        )
        self.consola.print(tabla)

        estado_meta = self.config_meta.estado_canales()
        tabla_meta = Table(
            title="Estado de Conectores Meta Platforms",
            show_header=True,
            header_style="bold magenta",
        )
        tabla_meta.add_column("Canal / Servicio", style="white")
        tabla_meta.add_column("Estado Credenciales (.env)", style="bold")
        for canal, listo in estado_meta.items():
            tabla_meta.add_row(
                canal,
                "[green]CONFIGURADO (Envio real activo)[/green]"
                if listo
                else "[yellow]PENDIENTE TOKEN (.env)[/yellow]",
            )
        self.consola.print(tabla_meta)

    async def _despachar_y_esperar(
        self,
        tipo_evento: str,
        payload: Dict[str, Any],
        conversation_id: str,
        esperar_coordinador: bool = False,
        timeout: float = 12.0,
    ) -> Dict[str, Any]:
        """Dispara un eventoLlegadaMensaje y espera el termino del ciclo FIPA-ACL."""
        async with self._lock_flujo:
            self.ventas.reiniciar_evento_ciclo()
            self.coordinador.reiniciar_evento_ciclo()

            evento = self.entorno.eventoLlegadaMensaje(
                tipo_evento=tipo_evento, payload=payload
            )
            self.ventas.procesar_evento_desde_entorno(
                evento=evento, conversation_id=conversation_id
            )

            await asyncio.wait_for(
                self.ventas.evento_ciclo_completado.wait(), timeout=timeout
            )
            if esperar_coordinador:
                await asyncio.wait_for(
                    self.coordinador.evento_ciclo_completado.wait(), timeout=timeout
                )
            await asyncio.sleep(0.05)
            return dict(self.ventas.ultimo_resultado)

    async def procesar_mensaje_entrante_meta(
        self,
        canal: str,
        id_cliente: str,
        nombre_cliente: str,
        texto: str,
        tipo_evento_sugerido: Optional[str] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Punto de entrada real desde los webhooks de WhatsApp, Instagram o Messenger.
        """
        idx_inicio = len(self.monitor.historial_mensajes)
        analisis_previo = self.modelo_nlp.analizar_mensaje(
            texto, contexto_tipo=tipo_evento_sugerido
        )
        tipo_evento = tipo_evento_sugerido or analisis_previo["intencion"]
        esperar_coordinador = analisis_previo["tono_cliente"] == "hostil"

        payload: Dict[str, Any] = {
            "id_cliente": id_cliente,
            "nombre": nombre_cliente,
            "canal": canal,
            "texto": texto,
            **(metadata_extra or {}),
        }
        conv_id = f"meta-{canal.lower()}-{id_cliente}-{len(self.entorno.eventos_disparados) + 1}"
        resultado = await self._despachar_y_esperar(
            tipo_evento=tipo_evento,
            payload=payload,
            conversation_id=conv_id,
            esperar_coordinador=esperar_coordinador,
        )
        mensajes_turno = self.monitor.historial_mensajes[idx_inicio:]
        return {
            "canal": canal,
            "id_cliente": id_cliente,
            "nombre_cliente": nombre_cliente,
            "texto_recibido": texto,
            "nlp": analisis_previo,
            "resultado": resultado,
            "mensajes_fipa_turno": mensajes_turno,
            "alerta_humana": (
                dict(self.coordinador.ultimo_resultado)
                if esperar_coordinador
                else None
            ),
        }

    async def responder_como_humano(
        self, id_cliente: str, texto: str, canal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Accion Human-in-the-Loop: permite al profesional humano responder
        directamente al cliente a traves de WhatsApp, Instagram o Messenger.
        """
        perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(id_cliente)
        canal_destino = canal or str(perfil.get("canal", "WhatsApp"))
        perfil["historial_interacciones"].append(
            {"rol": "profesional_humano", "texto": texto}
        )
        self.entorno.plataformas.emitir_respuesta_canal(
            canal=canal_destino,
            id_cliente=id_cliente,
            accion="respuesta_directa_humano",
            contenido=texto,
        )
        envio_meta = await self.entorno.despachar_salida_meta(
            canal=canal_destino, id_cliente=id_cliente, texto=texto
        )
        self.monitor.registrar_evento_entorno(
            f"HumanInTheLoop -> Meta[{canal_destino}]",
            f"cliente={id_cliente} | '{texto}'",
        )
        return {
            "id_cliente": id_cliente,
            "canal": canal_destino,
            "texto": texto,
            "meta": envio_meta,
        }

    def reactivar_automatizacion(self, id_cliente: str) -> Dict[str, Any]:
        """Devuelve el control del chat de un cliente desde el humano al SMA."""
        perfil = self.entorno.perfilesClientes.actualizar_perfil(
            id_cliente, automatizacion_activa=True, tono_actual="neutral"
        )
        for alerta in self.entorno.alertas_humanas:
            if alerta.get("id_cliente") == id_cliente:
                alerta["resuelta"] = True
        self.coordinador.creencias["estadoAgentes"]["AgenteVentas"] = "activo"
        self.monitor.registrar_evento_entorno(
            "HumanInTheLoop -> SistemaMultiagenteComercial",
            f"Automatizacion reactivada para cliente={id_cliente}",
        )
        return perfil


# Alias compatible con referencias previas
SistemaComercialOrquestador = ServicioMultiagenteComercial
