"""
Servicio principal del SistemaMultiagenteComercial.
Ensambla el entorno EntornoVirtualAtencion conectado a Botpress (WhatsApp,
Webchat, Messenger e Instagram), la organizacion MAS-ML SistemaMultiagenteComercial,
los modelos de Scikit-Learn y los 5 agentes SPADE sobre XMPP embebido.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

from src.agentes.agente_agendamientos import AgenteAgendamientos
from src.agentes.agente_coordinador import AgenteCoordinador
from src.agentes.agente_cotizaciones import AgenteCotizaciones
from src.agentes.agente_promociones import AgentePromociones
from src.agentes.agente_ventas import AgenteVentas
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.comunicacion.spade_runtime import detener_agentes_spade, iniciar_agentes_spade
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.ml.modelos_ml import (
    ModeloCotizacionesRF,
    ModeloNLPVentas,
    ModeloPromocionesKNN,
)
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.persistencia.google_calendar import ClienteGoogleCalendar
from src.persistencia.repositorio import RepositorioEntorno, crear_repositorio


class ServicioMultiagenteComercial:
    """
    Orquestador de ejecucion real para el SistemaMultiagenteComercial
    conectado con Botpress Studio, Supabase PostgreSQL y Google Calendar API v3.
    """

    def __init__(
        self,
        config_botpress: Optional[ConfiguracionBotpress] = None,
        repositorio: Optional[RepositorioEntorno] = None,
        google_calendar: Optional[ClienteGoogleCalendar] = None,
        usar_xmpp_embebido: bool = True,
        consola: Optional[Console] = None,
        silencioso: bool = False,
    ) -> None:
        raiz = Path(__file__).resolve().parents[2]
        self.config_botpress = config_botpress or ConfiguracionBotpress.desde_entorno(
            raiz / ".env"
        )
        self.usar_xmpp_embebido = usar_xmpp_embebido
        self.consola = consola or Console()
        self.silencioso = silencioso

        # 1. Conexion a BD en la nube (Supabase) o respaldo SQLite local
        if repositorio is not None:
            self.repositorio = repositorio
        else:
            ruta_sqlite = (
                ":memory:"
                if self.silencioso and not self.config_botpress.usa_supabase
                else (raiz / self.config_botpress.sqlite_db_path)
            )
            self.repositorio = crear_repositorio(
                supabase_url=self.config_botpress.supabase_url,
                supabase_key=self.config_botpress.supabase_key,
                negocio_id=self.config_botpress.negocio_id,
                sqlite_path=ruta_sqlite,
            )

        # 2. Conexion a Google Calendar API v3 (si esta configurado)
        if google_calendar is not None:
            self.google_calendar: Optional[ClienteGoogleCalendar] = google_calendar
        elif self.config_botpress.usa_google_calendar:
            cred_valor = self.config_botpress.google_service_account_json.strip()
            if not cred_valor.startswith("{") and not Path(cred_valor).is_absolute():
                cred_valor = str(raiz / cred_valor)
            self.google_calendar = ClienteGoogleCalendar(
                calendar_id=self.config_botpress.google_calendar_id,
                service_account_json=cred_valor,
                timezone=self.config_botpress.google_timezone,
            )
        else:
            self.google_calendar = None

        # 3. Monitor FIPA-ACL y Entorno MAS-ML alimentado desde BD + Google Calendar
        self.monitor = MonitorFIPAACL(consola=self.consola, silencioso=self.silencioso)
        self.entorno = EntornoVirtualAtencion(
            logger_evento=self.monitor.registrar_evento_entorno,
            repositorio=self.repositorio,
            google_calendar=self.google_calendar,
        )
        self.df_tarifas = self.entorno.tarifasYMargenes.tarifas_historicas
        self.df_ofertas = self.entorno.tarifasYMargenes.historico_ofertas

        # 3. Entrenamiento dinamico de modelos Scikit-Learn con los datos del negocio
        catalogo = self.entorno.plataformas.portafolio_servicios
        unidad = str(self.entorno.config_negocio.get("unidad_cantidad", "horas"))
        self.modelo_nlp = ModeloNLPVentas(
            catalogo_servicios=catalogo,
            unidad_cantidad=unidad,
        )
        self.modelo_rf = ModeloCotizacionesRF(
            self.df_tarifas, catalogo_servicios=catalogo
        )
        self.modelo_knn = ModeloPromocionesKNN(self.df_ofertas)

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

    @staticmethod
    def _extraer_texto_respuesta(resultado: Dict[str, Any], accion: str) -> str:
        """Extrae el texto final emitido por AgenteVentas para devolverlo a Botpress."""
        if accion == "en_atencion_humana":
            return ""
        envio = resultado.get("envio") or resultado.get("respuesta")
        if isinstance(envio, dict) and envio.get("contenido"):
            return str(envio["contenido"])
        if accion == "automatizacion_pausada":
            return (
                "Tu caso fue escalado directamente con nuestro asesor encargado. "
                "En un momento te respondera por este mismo chat."
            )
        return "Solicitud procesada por el sistema multiagente."

    async def procesar_mensaje_entrante(
        self,
        canal: str,
        id_cliente: str,
        nombre_cliente: str,
        texto: str,
        tipo_evento_sugerido: Optional[str] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Punto de entrada principal para mensajes provenientes desde Botpress
        (WhatsApp, Webchat, Facebook Messenger o Instagram).
        """
        idx_inicio = len(self.monitor.historial_mensajes)
        analisis_previo = self.modelo_nlp.analizar_mensaje(
            texto, contexto_tipo=tipo_evento_sugerido
        )
        tipo_evento = tipo_evento_sugerido or analisis_previo["intencion"]
        perfil_previo = self.entorno.perfilesClientes.obtener_o_crear_perfil(
            id_cliente, nombre=nombre_cliente, canal=canal
        )
        estaba_pausado = not bool(perfil_previo.get("automatizacion_activa", True))
        esperar_coordinador = (
            not estaba_pausado and analisis_previo["tono_cliente"] == "hostil"
        )

        payload: Dict[str, Any] = {
            "id_cliente": id_cliente,
            "nombre": nombre_cliente,
            "canal": canal,
            "texto": texto,
            **(metadata_extra or {}),
        }
        conv_id = f"bp-{canal.lower()}-{id_cliente}-{len(self.entorno.eventos_disparados) + 1}"
        resultado = await self._despachar_y_esperar(
            tipo_evento=tipo_evento,
            payload=payload,
            conversation_id=conv_id,
            esperar_coordinador=esperar_coordinador,
        )
        mensajes_turno = self.monitor.historial_mensajes[idx_inicio:]
        perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(id_cliente)
        pausado_humano = not bool(perfil.get("automatizacion_activa", True))
        accion = str(
            resultado.get("resultado") or resultado.get("accion") or "procesado"
        )
        respuesta_texto = self._extraer_texto_respuesta(resultado, accion)

        alerta_humana = (
            dict(self.coordinador.ultimo_resultado)
            if esperar_coordinador
            else None
        )
        notificar_dueno = accion == "automatizacion_pausada"
        responder_al_cliente = accion != "en_atencion_humana"

        mensaje_notificacion_dueno = ""
        if notificar_dueno:
            motivo = str(
                (alerta_humana or {}).get("motivo_critico", "reclamo_critico_cliente")
            )
            mensaje_notificacion_dueno = (
                "ALERTA DE ATENCION HUMANA - SistemaMultiagenteComercial\n"
                f"Cliente: {perfil.get('nombre', nombre_cliente)}\n"
                f"ID / Telefono: {id_cliente}\n"
                f"Canal: {canal}\n"
                f"Motivo de pausa: {motivo}\n"
                f"Ultimo mensaje: {texto}\n"
                "La automatizacion quedo pausada unicamente para este chat. "
                "Ingresa al WhatsApp corporativo para responder directamente."
            )

        telefono_dueno = (
            self.config_botpress.owner_whatsapp_number
            or str(self.entorno.config_negocio.get("telefono_dueno", ""))
        )

        return {
            "status": "ok",
            "canal": canal,
            "id_cliente": id_cliente,
            "nombre_cliente": nombre_cliente,
            "texto_recibido": texto,
            "respuesta": respuesta_texto,
            "responder_al_cliente": responder_al_cliente,
            "pausado_humano": pausado_humano,
            "notificar_dueno": notificar_dueno,
            "telefono_dueno": telefono_dueno,
            "mensaje_notificacion_dueno": mensaje_notificacion_dueno,
            "accion": accion,
            "intencion": analisis_previo["intencion"],
            "tono_cliente": analisis_previo["tono_cliente"],
            "nlp": analisis_previo,
            "resultado": resultado,
            "mensajes_fipa_turno": mensajes_turno,
            "alerta_humana": alerta_humana,
        }

    def reactivar_automatizacion(self, id_cliente: str) -> Dict[str, Any]:
        """Devuelve el control del chat de un cliente desde el humano al SMA."""
        perfil = self.entorno.perfilesClientes.actualizar_perfil(
            id_cliente, automatizacion_activa=True, tono_actual="neutral"
        )
        for alerta in self.entorno.alertas_humanas:
            if alerta.get("id_cliente") == id_cliente:
                alerta["resuelta"] = True
        self.entorno.repositorio.resolver_alertas_cliente(id_cliente)
        self.coordinador.creencias["estadoAgentes"]["AgenteVentas"] = "activo"
        self.monitor.registrar_evento_entorno(
            "HumanInTheLoop -> SistemaMultiagenteComercial",
            f"Automatizacion reactivada para cliente={id_cliente}",
        )
        return perfil
