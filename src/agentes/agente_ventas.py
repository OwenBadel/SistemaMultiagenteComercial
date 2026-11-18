"""
Implementacion del AgenteVentas (<<play>> rol EjecutivoVentas).
Arquitectura: Con aprendizaje (NLP para intencion y sentimiento/tono del cliente:
interesado, indeciso u hostil) y basado en utilidad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

from src.agentes.base_agente import JID_COORDINADOR, AgenteBaseMASML
from src.comunicacion.fipa_acl import extraer_payload_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.ml.modelos_ml import ModeloNLPVentas
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial


@dataclass
class PlanAtencionInicial:
    """Plan MAS-ML para atención empática, envío de catálogo y perfilamiento."""

    nombre: str = "PlanAtencionInicial"
    objetivo: str = "Atender inicialmente en el chat y escalar casos fuera de dominio"


@dataclass
class PlanCierreConPromo:
    """Plan MAS-ML para presentar promociones autorizadas y guiar hacia la compra."""

    nombre: str = "PlanCierreConPromo"
    objetivo: str = "Comunicar persuasivamente incentivos aprobados por el Coordinador"


class AgenteVentas(AgenteBaseMASML):
    """
    <<agent>> AgenteVentas:
    Puerta única de entrada al cliente. Integra NLP con Scikit-Learn y coordina
    el escalamiento al AgenteCoordinador.
    """

    def __init__(
        self,
        entorno: EntornoVirtualAtencion,
        organizacion: SistemaMultiagenteComercial,
        monitor: MonitorFIPAACL,
        modelo_nlp: ModeloNLPVentas,
        jid: str = "ventas@localhost",
        password: str = "sma_2026",
    ) -> None:
        super().__init__(
            jid=jid,
            password=password,
            nombre_agente="AgenteVentas",
            nombre_rol="EjecutivoVentas",
            arquitectura_tipo="Con aprendizaje (NLP) y basado en utilidad",
            entorno=entorno,
            organizacion=organizacion,
            monitor=monitor,
        )
        self.modelo_nlp = modelo_nlp
        self.creencias: Dict[str, Any] = {
            "mensajesEntrantes": [],
            "historialCharla": [],
            "tonoCliente": "neutral",
            "promocionAprobada": None,
        }
        self.objetivos = [
            "Maximizar interes, satisfaccion y conversion",
        ]
        self.planes = {
            "PlanAtencionInicial": PlanAtencionInicial(),
            "PlanCierreConPromo": PlanCierreConPromo(),
        }

    class ComportamientoEscuchaFIPA(CyclicBehaviour):
        """Recibe respuestas del AgenteCoordinador mediante mensajes FIPA-ACL."""

        async def run(self) -> None:
            msg: Message | None = await self.receive(timeout=0.2)
            if msg is None:
                return
            await self.agent.procesar_respuesta_coordinador(self, msg)

    class ComportamientoEntradaEntorno(OneShotBehaviour):
        """Procesa un evento entrante disparado desde EntornoVirtualAtencion."""

        def __init__(self, evento: Dict[str, Any], conversation_id: str) -> None:
            super().__init__()
            self.evento = evento
            self.conversation_id = conversation_id

        async def run(self) -> None:
            await self.agent.atender_evento_entorno(
                comportamiento=self,
                evento=self.evento,
                conversation_id=self.conversation_id,
            )

    async def setup(self) -> None:
        self.add_behaviour(
            self.ComportamientoEscuchaFIPA(), self.crear_plantilla_comercial()
        )

    def procesar_evento_desde_entorno(
        self, evento: Dict[str, Any], conversation_id: str
    ) -> None:
        """Lanza un OneShotBehaviour en SPADE para atender un eventoLlegadaMensaje."""
        self.reiniciar_evento_ciclo()
        comportamiento = self.ComportamientoEntradaEntorno(
            evento=evento, conversation_id=conversation_id
        )
        self.add_behaviour(comportamiento)

    # ------------------------------------------------------------------
    # ACCIONES DEL DIAGRAMA DE CLASES MAS-ML
    # ------------------------------------------------------------------
    def interpretarLenguajeNatural(
        self, texto: str, tipo_evento: Optional[str] = None
    ) -> Dict[str, Any]:
        """Clasifica intención y detecta tonoCliente usando ModeloNLPVentas."""
        analisis = self.modelo_nlp.analizar_mensaje(texto=texto, contexto_tipo=tipo_evento)
        self.creencias["mensajesEntrantes"].append(texto)
        self.creencias["historialCharla"].append({"rol": "cliente", "texto": texto})
        self.creencias["tonoCliente"] = analisis["tono_cliente"]
        self.registrar_accion(
            "interpretarLenguajeNatural()",
            f"intencion={analisis['intencion']} | tonoCliente={analisis['tono_cliente']}",
        )
        return analisis

    def enviarCatalogoVisual(self, canal: str, id_cliente: str) -> Dict[str, Any]:
        """Envia el portafolio visual de servicios cargado dinamicamente desde BD."""
        cfg = self.entorno.config_negocio
        catalogo = self.entorno.plataformas.portafolio_servicios
        intro = str(
            cfg.get(
                "mensaje_catalogo_intro",
                "Con gusto te comparto nuestro portafolio de servicios",
            )
        )
        unidad = str(cfg.get("unidad_cantidad", "unidades"))
        moneda = str(cfg.get("moneda", "COP"))
        lineas = [
            f"- {meta.get('nombre', clave)} (desde ${float(meta.get('tarifa_unitaria', meta.get('tarifa_hora_cop', 0))):,.0f} {moneda}/{unidad})"
            for clave, meta in catalogo.items()
        ]
        texto_catalogo = (
            f"Hola! {intro}:\n"
            + "\n".join(lineas)
            + f"\nIndicame que servicio necesitas y la cantidad de {unidad} (y complejidad 1-5 si aplica)."
        )
        self.creencias["historialCharla"].append(
            {"rol": "agente_ventas", "texto": texto_catalogo}
        )
        self.registrar_accion("enviarCatalogoVisual(portafolio_servicios)", f"canal={canal}")
        res = self.entorno.registrarRespuestaEntorno(
            accion="enviarCatalogoVisual(portafolio_servicios)",
            canal=canal,
            id_cliente=id_cliente,
            contenido=texto_catalogo,
        )
        return res

    def responderLenguajeNatural(
        self, tipo_respuesta: str, canal: str, id_cliente: str, mensaje: str
    ) -> Dict[str, Any]:
        """Envía respuesta natural al cliente mediante EntornoVirtualAtencion."""
        self.creencias["historialCharla"].append(
            {"rol": "agente_ventas", "texto": mensaje}
        )
        self.registrar_accion(f"responderLenguajeNatural({tipo_respuesta})", f"canal={canal}")
        res = self.entorno.registrarRespuestaEntorno(
            accion=f"responderLenguajeNatural({tipo_respuesta})",
            canal=canal,
            id_cliente=id_cliente,
            contenido=mensaje,
        )
        return res

    def presentarPromocionAlCliente(
        self, descuento_porciento: float, canal: str, id_cliente: str, valor_final: float
    ) -> Dict[str, Any]:
        """Comunica al cliente la promocion previamente autorizada por el Coordinador."""
        moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
        self.creencias["promocionAprobada"] = descuento_porciento
        mensaje = (
            f"Tenemos una oferta especial autorizada para ti: {descuento_porciento:.0f}% de descuento, "
            f"quedando el valor final en ${valor_final:,.0f} {moneda}. Agendamos tu cita de inicio?"
        )
        self.creencias["historialCharla"].append(
            {"rol": "agente_ventas", "texto": mensaje}
        )
        pct_tag = f"descuento_{int(round(descuento_porciento))}_porciento"
        self.registrar_accion(f"presentarPromocionAlCliente({pct_tag})", f"canal={canal}")
        return self.entorno.registrarRespuestaEntorno(
            accion=f"presentarPromocionAlCliente({pct_tag})",
            canal=canal,
            id_cliente=id_cliente,
            contenido=mensaje,
        )

    async def escalarCasoACoordinador(
        self,
        comportamiento: OneShotBehaviour,
        performativa: str,
        predicado: str,
        protocolo: str,
        conversation_id: str,
        payload_extra: Dict[str, Any],
    ) -> None:
        """Escala formalmente el caso hacia el AgenteCoordinador vía FIPA-ACL."""
        self.registrar_accion("escalarCasoACoordinador()", predicado)
        await self.enviar_fipa(
            comportamiento,
            JID_COORDINADOR,
            performativa,
            predicado,
            protocolo=protocolo,
            conversation_id=conversation_id,
            payload_extra=payload_extra,
        )

    # ------------------------------------------------------------------
    # LÓGICA DE ATENCIÓN AL ENTORNO Y RESPUESTAS FIPA DEL COORDINADOR
    # ------------------------------------------------------------------
    async def atender_evento_entorno(
        self,
        comportamiento: OneShotBehaviour,
        evento: Dict[str, Any],
        conversation_id: str,
    ) -> None:
        payload_evt = dict(evento.get("payload", {}))
        tipo_evento = str(evento.get("tipo_evento", ""))
        texto = str(payload_evt.get("texto", ""))
        canal = str(payload_evt.get("canal", "WhatsApp"))
        id_cliente = str(payload_evt.get("id_cliente", "cliente_001"))

        perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(
            id_cliente, canal=canal
        )
        # Si el caso fue escalado al profesional humano (Human-in-the-Loop), el bot permanece pausado
        if not perfil.get("automatizacion_activa", True):
            self.registrar_accion(
                "atencionPausadaHumanInTheLoop()",
                f"cliente={id_cliente} | canal={canal}",
            )
            self.marcar_ciclo_completado(
                {"resultado": "en_atencion_humana", "id_cliente": id_cliente, "canal": canal}
            )
            return

        analisis = self.interpretarLenguajeNatural(texto=texto, tipo_evento=tipo_evento)
        self.entorno.perfilesClientes.actualizar_perfil(
            id_cliente, tono_actual=analisis["tono_cliente"]
        )

        # Caso 4: Tono hostil o reclamo critico -> alerta_caso_critico
        if analisis["tono_cliente"] == "hostil" or analisis["intencion"] == "reclamo_critico":
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="INFORM",
                predicado="(alerta_caso_critico tono_hostil)",
                protocolo="FIPA-Request",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": canal,
                    "motivo_critico": texto,
                    "tono_cliente": "hostil",
                },
            )
            return

        # Si el cliente venia de parametros incompletos y responde con datos faltantes
        parametros_extraidos = dict(analisis.get("parametros_extraidos", {}))
        esperando_params = bool(perfil.get("esperando_parametros_cotizacion", False))
        tiene_nuevos_params = any(
            k in parametros_extraidos
            for k in ("tipo_servicio", "horas_estimadas", "complejidad", "urgencia_dias")
        )

        # Paso 2 (Cotizacion): Solicitud de cotizacion o respuesta aclaratoria con parametros
        if (
            tipo_evento == "parametros_servicio"
            or analisis["intencion"] == "solicitud_cotizacion"
            or (esperando_params and tiene_nuevos_params)
        ):
            params_previos = dict(perfil.get("parametros_parciales", {}))
            params = {
                **params_previos,
                **parametros_extraidos,
                **dict(payload_evt.get("parametros_servicio", {})),
            }
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="REQUEST",
                predicado="(escalar_caso cotizacion parametros_servicio)",
                protocolo="FIPA-Request",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": canal,
                    "parametros_servicio": params,
                },
            )
            return

        # Paso 1 (Atencion Inicial): Consulta de servicio -> envia catalogo visual
        if tipo_evento == "consulta_servicio" or analisis["intencion"] == "consulta_servicio":
            res = self.enviarCatalogoVisual(canal=canal, id_cliente=id_cliente)
            self.marcar_ciclo_completado(
                {"accion": "catalogo_enviado", "respuesta": res}
            )
            return

        # Caso 2: Objecion de precio / cliente indeciso -> solicita evaluar incentivo
        if tipo_evento == "objecion_precio" or analisis["intencion"] == "objecion_precio":
            nivel_indecision = float(
                payload_evt.get("nivel_indecision", analisis["nivel_indecision"])
            )
            cotizacion_actual = float(
                payload_evt.get("cotizacion_actual")
                or perfil.get("ultima_cotizacion")
                or 4_500_000.0
            )
            complejidad = int(
                payload_evt.get("complejidad")
                or perfil.get("ultima_complejidad")
                or 3
            )
            desc_solicitado = float(
                parametros_extraidos.get("descuento_solicitado_cliente", 0.0)
            )
            fidelidad = int(payload_evt.get("fidelidad", perfil.get("fidelidad", 1)))
            if desc_solicitado >= 18.0:
                nivel_indecision = max(nivel_indecision, 0.96)
                fidelidad = 3

            pred = (
                "(evaluar_incentivo nivel_indecision_alto)"
                if nivel_indecision >= 0.65
                else "(evaluar_incentivo nivel_indecision_medio)"
            )
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="REQUEST",
                predicado=pred,
                protocolo="FIPA-Propose",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": canal,
                    "nivel_indecision": nivel_indecision,
                    "cotizacion_actual": cotizacion_actual,
                    "fidelidad_cliente": fidelidad,
                    "complejidad": complejidad,
                    "descuento_solicitado_cliente": desc_solicitado,
                },
            )
            return

        # Caso 3: Aceptacion y solicitud de agendamiento
        if tipo_evento == "aceptar_cotizacion" or analisis["intencion"] == "solicitud_agendamiento":
            fecha_hora = str(
                payload_evt.get("fecha_hora")
                or parametros_extraidos.get("fecha_hora", "2026-11-18 10:00")
            )
            cotizacion_aprobada = bool(
                payload_evt.get("cotizacion_aprobada", perfil.get("ultima_cotizacion") is not None or True)
            )
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="REQUEST",
                predicado="(escalar_caso agendamiento fecha_hora)",
                protocolo="FIPA-Query",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "nombre_cliente": payload_evt.get("nombre", perfil.get("nombre", "Cliente Empresarial")),
                    "canal": canal,
                    "fecha_hora": fecha_hora,
                    "cotizacion_aprobada": cotizacion_aprobada,
                },
            )
            return

        # Fallback cordial si la intencion no encaja en categorias previas
        res = self.enviarCatalogoVisual(canal=canal, id_cliente=id_cliente)
        self.marcar_ciclo_completado(
            {"accion": "catalogo_enviado", "respuesta": res}
        )

    async def procesar_respuesta_coordinador(
        self, comportamiento: CyclicBehaviour, msg: Message
    ) -> None:
        perf = (msg.get_metadata("performative") or "").upper()
        conv_id = msg.get_metadata("conversation_id") or msg.thread or "conv-default"
        contenido = msg.body or ""
        payload = extraer_payload_fipa(msg)
        canal = str(payload.get("canal", "WhatsApp"))
        id_cliente = str(payload.get("id_cliente", "cliente_001"))

        if perf == "AGREE":
            # El Coordinador aceptó delegar el caso al especialista
            return

        if perf == "INFORM" and "entregar_cotizacion" in contenido:
            valor_total = float(payload.get("valor_total", 0.0))
            desglose = payload.get("desglose", {})
            moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
            costo_base = float(
                desglose.get("costo_base", desglose.get("costo_ingenieria_base_cop", 0))
            )
            ajuste = float(
                desglose.get(
                    "ajuste_complejidad_urgencia",
                    desglose.get("ajuste_complejidad_urgencia_cop", 0),
                )
            )
            texto_cot = (
                f"Tu cotizacion formal estimada para '{desglose.get('servicio', 'Servicio')}' "
                f"es de ${valor_total:,.0f} {moneda} (Base: ${costo_base:,.0f} {moneda}, "
                f"Ajuste por alcance/urgencia: ${ajuste:,.0f} {moneda})."
            )
            res = self.responderLenguajeNatural("cotizacion_formal", canal, id_cliente, texto_cot)
            self.marcar_ciclo_completado(
                {"resultado": "cotizacion_entregada", "valor_total": valor_total, "desglose": desglose, "envio": res}
            )

        elif perf == "REQUEST" and "solicitar_parametros_faltantes" in contenido:
            etiqueta_default = str(
                self.entorno.config_negocio.get("etiqueta_cantidad", "cantidad_unidades")
            )
            faltantes = payload.get("faltantes", [etiqueta_default])
            texto_aclaracion = (
                "Para entregarte una cotización exacta necesito confirmar los siguientes "
                f"parámetros obligatorios: {', '.join(faltantes)}."
            )
            res = self.responderLenguajeNatural("pregunta_aclaratoria", canal, id_cliente, texto_aclaracion)
            self.marcar_ciclo_completado(
                {"resultado": "parametros_incompletos", "faltantes": faltantes, "envio": res}
            )

        elif perf == "INFORM" and "transferir_promocion" in contenido:
            descuento = float(payload.get("descuento_porciento", 10.0))
            valor_final = float(payload.get("valor_con_descuento", 4_050_000.0))
            res = self.presentarPromocionAlCliente(
                descuento_porciento=descuento,
                canal=canal,
                id_cliente=id_cliente,
                valor_final=valor_final,
            )
            self.marcar_ciclo_completado(
                {"resultado": "promocion_presentada", "descuento_porciento": descuento, "valor_final": valor_final, "envio": res}
            )

        elif perf == "REFUSE" and "descuento_no_viable" in contenido:
            texto_refuse = (
                "En este momento no es posible aplicar un descuento adicional sin afectar la "
                "calidad del servicio, por lo que mantenemos la tarifa base formal."
            )
            res = self.responderLenguajeNatural("mantener_tarifa_base", canal, id_cliente, texto_refuse)
            self.marcar_ciclo_completado(
                {"resultado": "descuento_rechazado", "envio": res}
            )

        elif perf == "INFORM" and "reserva_confirmada" in contenido:
            fecha_hora = str(payload.get("fecha_hora", ""))
            texto_cita = f"Cita confirmada exitosamente para el {fecha_hora}. Te hemos enviado el recordatorio."
            res = self.responderLenguajeNatural("confirmacion_cita", canal, id_cliente, texto_cita)
            self.marcar_ciclo_completado(
                {"resultado": "cita_confirmada", "fecha_hora": fecha_hora, "envio": res}
            )

        elif perf == "INFORM" and "ofrecer_alternativas" in contenido:
            alternativas = payload.get("alternativas", [])
            texto_alt = (
                "El horario solicitado ya se encuentra ocupado en nuestra agenda. "
                f"Te propongo estos bloques disponibles: {', '.join(alternativas)}."
            )
            res = self.responderLenguajeNatural("sugerir_horarios_alternativos", canal, id_cliente, texto_alt)
            self.marcar_ciclo_completado(
                {"resultado": "horarios_alternativos", "alternativas": alternativas, "envio": res}
            )

        elif perf == "CANCEL" and "cancelar_accion" in contenido:
            self.registrar_accion("pausarChatAutomatico()", f"cliente={id_cliente}")
            await self.enviar_fipa(
                comportamiento,
                JID_COORDINADOR,
                "CONFIRM",
                "(automatizacion_pausada chat_cliente)",
                protocolo="FIPA-Request",
                conversation_id=conv_id,
                payload_extra={"id_cliente": id_cliente, "canal": canal},
            )
            self.marcar_ciclo_completado({"resultado": "automatizacion_pausada", "id_cliente": id_cliente})

