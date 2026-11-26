"""
Implementacion del AgenteVentas (<<play>> rol EjecutivoVentas).
Arquitectura: Con aprendizaje (NLP para intencion y sentimiento/tono del cliente:
interesado, indeciso u hostil) y basado en utilidad.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

from src.agentes.base_agente import JID_COORDINADOR, AgenteBaseMASML
from src.comunicacion.fipa_acl import extraer_payload_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.ml.generador_llm import GeneradorDialogoGemini
from src.ml.modelos_ml import (
    ModeloNLPVentas,
    _es_objecion_precio_explicita,
    normalizar_unidad_comercial,
)
from src.organizacion.organizacion_comercial import SistemaMultiagenteComercial
from src.reglas_comerciales import promocion_elegible, validar_datos_cliente_reserva


def _es_respuesta_afirmativa(texto: str) -> bool:
    normalizado = unicodedata.normalize("NFD", texto.casefold())
    normalizado = "".join(
        caracter
        for caracter in normalizado
        if unicodedata.category(caracter) != "Mn"
    )
    if re.search(r"\bno\b", normalizado):
        return False
    return bool(
        re.search(
            r"\b(?:si|claro|perfecto|de acuerdo|hagamoslo|por favor)\b",
            normalizado,
        )
    )


def _nombre_completo_valido(texto: str) -> bool:
    nombre = " ".join(texto.split())
    normalizado = unicodedata.normalize("NFD", nombre.casefold())
    normalizado = "".join(
        caracter
        for caracter in normalizado
        if unicodedata.category(caracter) != "Mn"
    )
    tokens = re.findall(r"[^\W\d_]+", nombre, flags=re.UNICODE)
    return len(tokens) >= 2 and normalizado not in {
        "cliente",
        "cliente potencial",
        "cliente empresarial",
        "anonymous user",
    }


def _es_aceptacion_explicita_cotizacion(texto: str) -> bool:
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    texto_normalizado = "".join(
        caracter
        for caracter in descompuesto
        if unicodedata.category(caracter) != "Mn"
    )
    if re.search(r"\bno\s+(?:acepto|apruebo|confirmo)\b", texto_normalizado):
        return False
    patrones = (
        r"\b(?:acepto|apruebo|confirmo)\s+(?:la\s+)?"
        r"(?:cotizacion|propuesta|oferta)\b",
        r"\bestoy\s+de\s+acuerdo\s+con\s+(?:la\s+)?"
        r"(?:cotizacion|propuesta|oferta)\b",
        r"\b(?:acepto|apruebo)\s+(?:el\s+)?(?:precio|presupuesto)\b",
        r"\b(?:cotizacion|propuesta|oferta)\s+(?:esta\s+)?"
        r"(?:aceptada|aprobada|confirmada)\b",
    )
    return any(re.search(patron, texto_normalizado) for patron in patrones)


def _es_rechazo_horario(texto: str) -> bool:
    descompuesto = unicodedata.normalize("NFD", texto.casefold())
    normalizado = "".join(
        caracter
        for caracter in descompuesto
        if unicodedata.category(caracter) != "Mn"
    )
    return bool(
        re.search(
            r"\bno\s+me\s+(?:sirve|conviene)\b.{0,24}\b(?:hora|horario|opcion|alternativa)\b"
            r"|\b(?:hora|horario|opcion|alternativa)\b.{0,24}\bno\s+me\s+(?:sirve|conviene)\b"
            r"|\bno\s+me\s+(?:sirve|conviene)\b.{0,24}\b(?:las?\s+)?\d{1,2}\b"
            r"|\bningun[oa]s?\s+de\s+(?:esas|esos)\s+(?:horas|horarios|opciones|alternativas)\b",
            normalizado,
        )
    )


def _fecha_hora_ya_paso(valor: str) -> bool:
    try:
        fecha_hora = datetime.fromisoformat(valor.strip().replace("Z", "+00:00"))
    except ValueError:
        return True
    zona = ZoneInfo("America/Bogota")
    if fecha_hora.tzinfo is None:
        fecha_hora = fecha_hora.replace(tzinfo=zona)
    else:
        fecha_hora = fecha_hora.astimezone(zona)
    return fecha_hora <= datetime.now(zona)


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
        generador_llm: Optional[GeneradorDialogoGemini] = None,
        jid: str = "ventas@localhost",
        password: str = "sma_2026",
    ) -> None:
        super().__init__(
            jid=jid,
            password=password,
            nombre_agente="AgenteVentas",
            nombre_rol="EjecutivoVentas",
            arquitectura_tipo="Con aprendizaje (NLP + LLM NLG) y basado en utilidad",
            entorno=entorno,
            organizacion=organizacion,
            monitor=monitor,
        )
        self.modelo_nlp = modelo_nlp
        self.generador_llm = generador_llm
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
        self,
        texto: str,
        tipo_evento: Optional[str] = None,
        campos_pendientes: Optional[list[str]] = None,
        parametros_acumulados: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Clasifica intención y detecta tonoCliente usando ModeloNLPVentas."""
        analisis = self.modelo_nlp.analizar_mensaje(
            texto=texto,
            contexto_tipo=tipo_evento,
            campos_pendientes=campos_pendientes,
            parametros_acumulados=parametros_acumulados,
        )
        self.creencias["mensajesEntrantes"].append(texto)
        self.creencias["historialCharla"].append({"rol": "cliente", "texto": texto})
        self.creencias["tonoCliente"] = analisis["tono_cliente"]
        self.registrar_accion(
            "interpretarLenguajeNatural()",
            f"intencion={analisis['intencion']} | tonoCliente={analisis['tono_cliente']}",
        )
        return analisis

    def _redactar_con_llm(
        self,
        tipo_respuesta: str,
        texto_base: str,
        hechos_obligatorios: Optional[list[str]] = None,
        id_cliente: Optional[str] = None,
    ) -> str:
        """Usa Gemini para redactar naturalmente conservando hechos verificados y fallback."""
        if self.generador_llm is None or not self.generador_llm.activo:
            return texto_base
        ultimo_msg = (
            str(self.creencias["mensajesEntrantes"][-1])
            if self.creencias.get("mensajesEntrantes")
            else ""
        )
        historial_reciente: list[Dict[str, Any]] = []
        if id_cliente and tipo_respuesta != "saludo_bienvenida":
            perfil_cli = self.entorno.perfilesClientes.perfiles.get(id_cliente, {})
            for item in perfil_cli.get("historial_interacciones", [])[-6:]:
                txt = str(item.get("texto") or item.get("contenido") or "").strip()
                if txt:
                    historial_reciente.append({"rol": item.get("rol", "cliente"), "texto": txt})
        if not historial_reciente and tipo_respuesta != "saludo_bienvenida":
            historial_reciente = list(self.creencias.get("historialCharla", []))
        return self.generador_llm.redactar_respuesta(
            tipo_respuesta=tipo_respuesta,
            texto_base=texto_base,
            contexto_negocio=self.entorno.config_negocio,
            mensaje_cliente=ultimo_msg,
            historial_reciente=historial_reciente,
            hechos_obligatorios=hechos_obligatorios,
        )

    @staticmethod
    def _describir_parametro_faltante(campo: str, unidad: str) -> str:
        mapa = {
            "tipo_servicio": "qué servicio necesitas",
            "horas_estimadas": "cuántas horas aproximadas estimas",
            "sesiones_requeridas": "cuántas sesiones necesitas",
            "cantidad_unidades": f"la cantidad de {unidad} que necesitas",
            "complejidad": "el nivel de alcance o detalle del proyecto",
        }
        return mapa.get(campo, campo.replace("_", " "))

    @staticmethod
    def _es_saludo_simple(texto: str) -> bool:
        """Retorna True si el mensaje es solo un saludo sin pedir paquetes ni servicios."""
        t = (texto or "").strip().casefold()
        if not t:
            return False
        palabras_catalogo = (
            "servicio", "paquete", "plan", "catalogo", "catálogo",
            "portafolio", "ofrecen", "tienen", "hacen", "producto",
            "lista", "opciones", "precio", "costo", "cuesta",
        )
        if any(p in t for p in palabras_catalogo):
            return False
        saludos = (
            "hola", "buenos dias", "buenos días", "buenas tardes",
            "buenas noches", "buenas", "hey", "que tal", "qué tal",
            "saludos", "holi", "buen dia", "buen día",
        )
        return any(s in t for s in saludos) and len(t.split()) <= 6

    def enviarCatalogoVisual(self, canal: str, id_cliente: str) -> Dict[str, Any]:
        """Envia bienvenida de negocio o paquetes sin precios (los precios los da AgenteCotizaciones)."""
        cfg = self.entorno.config_negocio
        catalogo = self.entorno.plataformas.portafolio_servicios
        nombre_negocio = str(cfg.get("nombre_negocio", "nuestro estudio"))
        rubro = str(cfg.get("rubro", "servicios profesionales"))
        unidad = str(cfg.get("unidad_cantidad", "unidades"))
        lineas_sin_precio = [
            f"- {meta.get('nombre', clave)}"
            for clave, meta in catalogo.items()
        ]
        ultimo_msg = (
            str(self.creencias["mensajesEntrantes"][-1])
            if self.creencias.get("mensajesEntrantes")
            else ""
        )
        ultimo_msg_cf = ultimo_msg.casefold()
        perfil_cli = self.entorno.perfilesClientes.perfiles.get(id_cliente, {})
        historial_cli = perfil_cli.get("historial_interacciones", [])
        ya_vio_catalogo = any(
            t.get("rol") == "agente_ventas" and "\n- " in str(t.get("contenido", ""))
            for t in historial_cli
        )
        pide_algo_especifico_o_no_esta = any(
            p in ultimo_msg_cf
            for p in ("no esta", "no está", "ninguno", "otra cosa", "familiar", "familia", "boda", "quince")
        )

        if self._es_saludo_simple(ultimo_msg):
            tipo_nlg = "saludo_bienvenida"
            texto_base = (
                f"¡Hola! Bienvenido a {nombre_negocio}, nos especializamos en {rubro}. "
                "¿En qué podemos ayudarte hoy?"
            )
            texto_base_nlg = texto_base
            hechos_catalogo = [nombre_negocio]
        elif (
            self.generador_llm is not None
            and self.generador_llm.activo
            and (ya_vio_catalogo or pide_algo_especifico_o_no_esta)
        ):
            tipo_nlg = "seguimiento_consulta_especifica"
            hechos_catalogo = []
            texto_base = (
                f"En {nombre_negocio} contamos con estos paquetes:\n"
                + "\n".join(lineas_sin_precio)
                + f"\nCuéntame cuál te interesa y cuántas {unidad} necesitas para prepararte una cotización."
            )
            texto_base_nlg = (
                f"Referencia de paquetes de {nombre_negocio} (SIN dar precios, los precios solo los calcula cotizaciones):\n"
                + "\n".join(lineas_sin_precio)
                + "\nResponde directamente a la necesidad del cliente sin inventar precios ni repetir mecanicamente la lista, "
                "y sugiere como adaptar alguno de nuestros paquetes o preguntale detalles para cotizarlo."
            )
        else:
            tipo_nlg = "catalogo_servicios"
            texto_base = (
                f"Con gusto, en {nombre_negocio} contamos con los siguientes paquetes:\n"
                + "\n".join(lineas_sin_precio)
                + f"\nCuéntame cuál te interesa y cuántas {unidad} planeas para que nuestro sistema te prepare una cotización."
            )
            hechos_catalogo = [
                str(meta.get("nombre", clave)) for clave, meta in catalogo.items()
            ]
            texto_base_nlg = texto_base

        texto_catalogo = self._redactar_con_llm(
            tipo_respuesta=tipo_nlg,
            texto_base=texto_base_nlg,
            hechos_obligatorios=hechos_catalogo,
            id_cliente=id_cliente,
        )
        if texto_catalogo == texto_base_nlg and tipo_nlg == "seguimiento_consulta_especifica":
            texto_catalogo = texto_base
        self.creencias["historialCharla"].append(
            {"rol": "agente_ventas", "texto": texto_catalogo}
        )
        if tipo_nlg == "saludo_bienvenida":
            accion_entorno = "responderLenguajeNatural(saludo_bienvenida)"
        elif tipo_nlg == "seguimiento_consulta_especifica":
            accion_entorno = "responderLenguajeNatural(asesoria_consulta)"
        else:
            accion_entorno = "enviarCatalogoVisual(portafolio_servicios)"

        self.registrar_accion(accion_entorno, f"canal={canal}")
        res = self.entorno.registrarRespuestaEntorno(
            accion=accion_entorno,
            canal=canal,
            id_cliente=id_cliente,
            contenido=texto_catalogo,
        )
        return res

    def responderLenguajeNatural(
        self,
        tipo_respuesta: str,
        canal: str,
        id_cliente: str,
        mensaje: str,
        hechos_obligatorios: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Envía respuesta natural al cliente mediante EntornoVirtualAtencion."""
        mensaje_final = self._redactar_con_llm(
            tipo_respuesta=tipo_respuesta,
            texto_base=mensaje,
            hechos_obligatorios=hechos_obligatorios,
            id_cliente=id_cliente,
        )
        self.creencias["historialCharla"].append(
            {"rol": "agente_ventas", "texto": mensaje_final}
        )
        self.registrar_accion(f"responderLenguajeNatural({tipo_respuesta})", f"canal={canal}")
        res = self.entorno.registrarRespuestaEntorno(
            accion=f"responderLenguajeNatural({tipo_respuesta})",
            canal=canal,
            id_cliente=id_cliente,
            contenido=mensaje_final,
        )
        return res

    def presentarPromocionAlCliente(
        self,
        descuento_porciento: float,
        canal: str,
        id_cliente: str,
        valor_final: float,
        conversation_id: Optional[str] = None,
        codigo_promocion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Vuelve a validar cotización y regla antes de comunicar una promoción."""
        self.creencias["promocionAprobada"] = None
        contexto = self.entorno.perfilesClientes.obtener_estado_conversacion(
            id_cliente, conversation_id
        )
        perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(id_cliente)
        parametros = self.entorno.perfilesClientes.obtener_parametros_cotizacion(
            id_cliente, conversation_id
        )
        valor_cotizado = (
            contexto.get("ultima_cotizacion")
            if conversation_id
            else perfil.get("ultima_cotizacion")
        )
        promocion = next(
            (
                item
                for item in self.entorno.tarifasYMargenes.promociones_vigentes
                if str(item.get("codigo", "")) == str(codigo_promocion or "")
            ),
            None,
        )
        cantidad = parametros.get("cantidad_unidades")
        elegible = bool(
            promocion
            and cantidad is not None
            and promocion_elegible(
                promocion,
                str(parametros.get("tipo_servicio", "")),
                cantidad,
                int(parametros["complejidad"])
                if parametros.get("complejidad") is not None
                else None,
                canal=canal,
            )
        )
        try:
            valor_cotizado_num = float(valor_cotizado)
            descuento = float(descuento_porciento)
            valor_final_esperado = round(
                valor_cotizado_num * (1.0 - descuento / 100.0), -2
            )
            valor_final_num = float(valor_final)
        except (TypeError, ValueError):
            valor_cotizado_num = 0.0
            descuento = 0.0
            valor_final_esperado = 0.0
            valor_final_num = -1.0
        limite_autorizado = min(
            float(promocion.get("descuento_max", 0)) if promocion else 0.0,
            self.entorno.tarifasYMargenes.descuento_maximo_permitido,
        )
        autorizada = (
            elegible
            and self.entorno.perfilesClientes.tiene_cotizacion(
                id_cliente, conversation_id
            )
            and not self.entorno.perfilesClientes.cotizacion_aprobada(
                id_cliente, conversation_id
            )
            and 0 < descuento <= limite_autorizado
            and round(valor_final_num, -2) == valor_final_esperado
        )
        if not autorizada:
            moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
            mensaje_base = (
                f"La cotización formal vigente es de ${valor_cotizado_num:,.0f} "
                f"{moneda}. Mantenemos esa tarifa porque no se pudo validar una "
                "promoción aplicable."
            )
            return self.responderLenguajeNatural(
                "mantener_tarifa_base", canal, id_cliente, mensaje_base
            )

        moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
        self.creencias["promocionAprobada"] = descuento
        mensaje_base = (
            f"Hay una promoción autorizada para esta cotización: {descuento:g}% "
            f"de descuento, con un valor final de ${valor_final_esperado:,.0f} "
            f"{moneda}. ¿Deseas continuar?"
        )
        mensaje = self._redactar_con_llm(
            tipo_respuesta="promocion_autorizada",
            texto_base=mensaje_base,
            hechos_obligatorios=[
                f"{descuento:g}%",
                f"${valor_final_esperado:,.0f}",
            ],
            id_cliente=id_cliente,
        )
        self.creencias["historialCharla"].append(
            {"rol": "agente_ventas", "texto": mensaje}
        )
        pct_tag = f"descuento_{descuento:g}_porciento"
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
        contexto_id_raw = payload_evt.get("contexto_conversacion_id")
        contexto_id = str(contexto_id_raw) if contexto_id_raw else None

        perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(
            id_cliente, canal=canal
        )
        # Si el caso fue escalado al profesional humano (Human-in-the-Loop), el bot permanece pausado
        if not perfil.get("automatizacion_activa", True):
            self.registrar_accion(
                "atencionPausadaHumanInTheLoop()",
                f"canal={canal}",
            )
            self.marcar_ciclo_completado(
                {"resultado": "en_atencion_humana", "id_cliente": id_cliente, "canal": canal}
            )
            return

        contexto_conversacion = (
            self.entorno.perfilesClientes.obtener_estado_conversacion(
                id_cliente, contexto_id
            )
        )
        estado_reserva = dict(contexto_conversacion.get("estado_reserva", {}))
        parametros_perfil = self.entorno.perfilesClientes.obtener_parametros_cotizacion(
            id_cliente, contexto_id
        )
        esperando_parametros = bool(
            contexto_conversacion.get(
                "esperando_parametros_cotizacion",
                perfil.get("esperando_parametros_cotizacion", False)
                if not contexto_id
                else False,
            )
        )
        campos_pendientes = (
            self.entorno.perfilesClientes.obtener_campos_cotizacion_pendientes(
                id_cliente, contexto_id
            )
            if esperando_parametros
            else []
        )
        if esperando_parametros and not campos_pendientes:
            campos_pendientes = self.modelo_nlp.campos_cotizacion_pendientes(
                parametros_perfil
            )
        contexto_nlp = tipo_evento
        if estado_reserva.get("paso") == "esperando_horario":
            contexto_nlp = "solicitud_agendamiento"
        analisis = self.interpretarLenguajeNatural(
            texto=texto,
            tipo_evento=contexto_nlp,
            campos_pendientes=campos_pendientes,
            parametros_acumulados=parametros_perfil,
        )
        self.entorno.perfilesClientes.actualizar_perfil(
            id_cliente, tono_actual=analisis["tono_cliente"]
        )

        # Solo señales verificadas o una predicción crítica suficientemente fuerte escalan.
        if analisis.get("escalamiento_humano", False):
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

        # Alternativa C: Cliente declina oferta y rechaza la compra
        if (
            tipo_evento in {"rechazo_compra", "rechazo_compra_no_interesado"}
            or analisis["intencion"] == "rechazo_compra"
        ):
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="INFORM",
                predicado="(cliente_declina_oferta motivo: excede_presupuesto)",
                protocolo="FIPA-Request",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": canal,
                    "contexto_conversacion_id": contexto_id,
                    "motivo_rechazo": texto or "excede_presupuesto",
                },
            )
            return

        if estado_reserva.get("paso") in {
            "nombre_cliente",
            "cedula",
            "telefono",
        }:
            paso = str(estado_reserva["paso"])
            if paso == "nombre_cliente":
                if not _nombre_completo_valido(texto):
                    mensaje = "Para la reserva necesito tu nombre y apellido. ¿Cuál es tu nombre completo?"
                    res = self.responderLenguajeNatural(
                        "nombre_completo_requerido", canal, id_cliente, mensaje
                    )
                    self.marcar_ciclo_completado(
                        {"resultado": "datos_cliente_pendientes", "envio": res}
                    )
                    return
                estado_reserva["nombre_cliente"] = " ".join(texto.split())
                estado_reserva["paso"] = "cedula"
                mensaje = "Gracias. ¿Cuál es tu número de identificación?"
            elif paso == "cedula":
                identificacion = re.sub(r"\D", "", texto)
                if not 5 <= len(identificacion) <= 15:
                    mensaje = "La identificación debe tener entre 5 y 15 dígitos. ¿Puedes verificarla?"
                    res = self.responderLenguajeNatural(
                        "identificacion_requerida", canal, id_cliente, mensaje
                    )
                    self.marcar_ciclo_completado(
                        {"resultado": "datos_cliente_pendientes", "envio": res}
                    )
                    return
                estado_reserva["cedula"] = identificacion
                estado_reserva["paso"] = "telefono"
                mensaje = "¿Cuál es tu teléfono de contacto, incluyendo el indicativo si aplica?"
            else:
                try:
                    datos_validos = validar_datos_cliente_reserva(
                        estado_reserva.get("nombre_cliente"),
                        estado_reserva.get("cedula"),
                        texto,
                    )
                except ValueError:
                    mensaje = "El teléfono debe tener entre 7 y 15 dígitos. ¿Puedes verificarlo?"
                    res = self.responderLenguajeNatural(
                        "telefono_requerido", canal, id_cliente, mensaje
                    )
                    self.marcar_ciclo_completado(
                        {"resultado": "datos_cliente_pendientes", "envio": res}
                    )
                    return
                estado_reserva.update(datos_validos)
                estado_reserva["paso"] = "esperando_horario"
                self.entorno.perfilesClientes.actualizar_estado_conversacion(
                    id_cliente, contexto_id, {"estado_reserva": estado_reserva}
                )
                fecha_pendiente = str(
                    parametros_perfil.get("fecha_agendamiento_pendiente", "")
                ).strip()
                if not fecha_pendiente:
                    fecha_pendiente = str(
                        parametros_perfil.get("fecha_hora", "")
                    ).strip()
                if fecha_pendiente and not _fecha_hora_ya_paso(fecha_pendiente):
                    tipo_evento = "solicitud_agendamiento"
                else:
                    fecha_solicitada = parametros_perfil.get(
                        "fecha_agendamiento_fecha"
                    )
                    mensaje = (
                        f"Ya tengo el día {fecha_solicitada}. ¿A qué hora te gustaría la cita?"
                        if fecha_solicitada
                        else "Tus datos quedaron registrados para esta conversación. ¿Qué día y hora te gustaría reservar?"
                    )
                    res = self.responderLenguajeNatural(
                        "fecha_agendamiento_requerida",
                        canal,
                        id_cliente,
                        mensaje,
                    )
                    self.marcar_ciclo_completado(
                        {
                            "resultado": "fecha_agendamiento_requerida",
                            "envio": res,
                        }
                    )
                    return
            self.entorno.perfilesClientes.actualizar_estado_conversacion(
                id_cliente, contexto_id, {"estado_reserva": estado_reserva}
            )
            if paso != "telefono":
                res = self.responderLenguajeNatural(
                    "datos_cliente_requeridos", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {"resultado": "datos_cliente_pendientes", "envio": res}
                )
                return

        # Si el cliente venia de parametros incompletos y responde con datos faltantes
        parametros_extraidos = dict(analisis.get("parametros_extraidos", {}))
        alternativas_ofrecidas = list(
            estado_reserva.get("alternativas_ofrecidas", [])
        )
        if (
            estado_reserva.get("paso") == "esperando_horario"
            and alternativas_ofrecidas
            and _es_rechazo_horario(texto)
        ):
            rechazadas = list(estado_reserva.get("alternativas_rechazadas", []))
            horario_rechazado = str(
                parametros_extraidos.get("fecha_hora", "")
            ).strip()
            para_rechazar = (
                [horario_rechazado]
                if horario_rechazado in alternativas_ofrecidas
                else alternativas_ofrecidas
            )
            estado_reserva["alternativas_rechazadas"] = list(
                dict.fromkeys([*rechazadas, *para_rechazar])
            )
            estado_reserva["alternativas_ofrecidas"] = []
            self.entorno.perfilesClientes.actualizar_estado_conversacion(
                id_cliente, contexto_id, {"estado_reserva": estado_reserva}
            )
            mensaje = "Entendido. ¿Qué otro día y horario te conviene?"
            res = self.responderLenguajeNatural(
                "horario_alternativo_rechazado", canal, id_cliente, mensaje
            )
            self.marcar_ciclo_completado(
                {"resultado": "horario_alternativo_rechazado", "envio": res}
            )
            return

        esperando_params = esperando_parametros
        tiene_nuevos_params = any(
            k in parametros_extraidos
            for k in (
                "tipo_servicio",
                "cantidad_unidades",
                "horas_estimadas",
                "sesiones_requeridas",
                "cantidad_productos",
                "cantidad_fotos",
                "cantidad_personas",
                "en_exterior",
                "es_personalizado",
                "complejidad",
                "urgencia_dias",
            )
        )

        # Paso 2 (Cotizacion): Solicitud de cotizacion o respuesta aclaratoria con parametros
        if (
            tipo_evento == "parametros_servicio"
            or analisis["intencion"] == "solicitud_cotizacion"
            or (esperando_params and tiene_nuevos_params)
        ):
            params_previos = dict(
                self.entorno.perfilesClientes.obtener_parametros_cotizacion(
                    id_cliente, contexto_id
                )
            )
            if (
                "tipo_servicio" in parametros_extraidos
                and not parametros_extraidos.get("es_personalizado")
                and "cantidad_fotos" not in parametros_extraidos
            ):
                for clave_custom in (
                    "es_personalizado",
                    "descripcion_personalizada",
                    "cantidad_fotos",
                    "cantidad_personas",
                    "en_exterior",
                ):
                    params_previos.pop(clave_custom, None)
            parametros_metadata = {
                clave: valor
                for clave, valor in dict(
                    payload_evt.get("parametros_servicio", {})
                ).items()
                if valor is not None and valor != ""
            }
            params = {
                **parametros_metadata,
                **params_previos,
                **parametros_extraidos,
            }
            fecha_hora_nueva = str(params.get("fecha_hora", "")).strip()
            if fecha_hora_nueva:
                if params.get("fecha_hora_pasada") or _fecha_hora_ya_paso(
                    fecha_hora_nueva
                ):
                    self.entorno.perfilesClientes.establecer_fecha_agendamiento_pendiente(
                        id_cliente, None, contexto_id
                    )
                else:
                    self.entorno.perfilesClientes.establecer_fecha_agendamiento_pendiente(
                        id_cliente, fecha_hora_nueva, contexto_id
                    )
            if esperando_params and tiene_nuevos_params:
                await self.escalarCasoACoordinador(
                    comportamiento=comportamiento,
                    performativa="INFORM",
                    predicado="(parametros_completados parametros_servicio)",
                    protocolo="FIPA-Request",
                    conversation_id=conversation_id,
                    payload_extra={
                        "id_cliente": id_cliente,
                        "canal": canal,
                        "contexto_conversacion_id": contexto_id,
                        "parametros_servicio": params,
                    },
                )
            else:
                await self.escalarCasoACoordinador(
                    comportamiento=comportamiento,
                    performativa="REQUEST",
                    predicado="(solicitar_cotizacion_servicio parametros_servicio)",
                    protocolo="FIPA-Request",
                    conversation_id=conversation_id,
                    payload_extra={
                        "id_cliente": id_cliente,
                        "canal": canal,
                        "contexto_conversacion_id": contexto_id,
                        "parametros_servicio": params,
                    },
                )
            return

        # Paso 1 (Atencion Inicial): Consulta de servicio -> envia catalogo visual
        if (
            estado_reserva.get("paso") != "esperando_horario"
            and (
                tipo_evento == "consulta_servicio"
                or analisis["intencion"] == "consulta_servicio"
            )
        ):
            res = self.enviarCatalogoVisual(canal=canal, id_cliente=id_cliente)
            self.marcar_ciclo_completado(
                {"accion": "catalogo_enviado", "respuesta": res}
            )
            return

        cotizacion_actual = self.entorno.perfilesClientes.obtener_estado_conversacion(
            id_cliente, contexto_id
        )
        valor_cotizacion = (
            cotizacion_actual.get("ultima_cotizacion")
            if contexto_id
            else perfil.get("ultima_cotizacion")
        )
        tiene_cotizacion = self.entorno.perfilesClientes.tiene_cotizacion(
            id_cliente, contexto_id
        )
        cotizacion_aprobada_actual = self.entorno.perfilesClientes.cotizacion_aprobada(
            id_cliente, contexto_id
        )

        # La predicción ML nunca basta para iniciar una negociación.
        if (
            _es_objecion_precio_explicita(texto)
            and (tipo_evento == "objecion_precio" or analisis["intencion"] == "objecion_precio")
        ):
            parametros_cliente = self.entorno.perfilesClientes.obtener_parametros_cotizacion(
                id_cliente, contexto_id
            )
            try:
                cotizacion_actual_num = float(valor_cotizacion)
            except (TypeError, ValueError):
                cotizacion_actual_num = 0.0
            servicio = str(parametros_cliente.get("tipo_servicio", ""))
            cantidad = parametros_cliente.get("cantidad_unidades")
            complejidad = parametros_cliente.get("complejidad")
            promociones_elegibles = [
                promo
                for promo in self.entorno.tarifasYMargenes.promociones_vigentes
                if promocion_elegible(
                    promo,
                    servicio,
                    cantidad,
                    int(complejidad) if complejidad is not None else None,
                    canal=canal,
                )
            ] if cantidad is not None else []

            if not tiene_cotizacion or cotizacion_actual_num <= 0:
                mensaje = (
                    "Para revisar condiciones comerciales primero necesitamos "
                    "preparar una cotización formal para el servicio."
                )
                res = self.responderLenguajeNatural(
                    "cotizacion_requerida", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {
                        "resultado": "cotizacion_requerida",
                        "envio": res,
                    }
                )
                return

            if cotizacion_aprobada_actual or not promociones_elegibles:
                moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
                mensaje = (
                    f"La cotización formal vigente es de ${cotizacion_actual_num:,.0f} "
                    f"{moneda}. No hay una promoción autorizada aplicable a este "
                    "servicio y cantidad; mantenemos la tarifa cotizada."
                )
                res = self.responderLenguajeNatural(
                    "mantener_tarifa_base", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {"resultado": "tarifa_base_mantenida", "envio": res}
                )
                return

            promocion = max(
                promociones_elegibles,
                key=lambda item: float(item.get("descuento_max", 0)),
            )
            nivel_indecision = float(analisis["nivel_indecision"])
            desc_solicitado = float(
                parametros_extraidos.get("descuento_solicitado_cliente", 0.0)
            )
            fidelidad = int(perfil.get("fidelidad", 1))
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="REQUEST",
                predicado="(evaluar_incentivo objecion_precio_verificada)",
                protocolo="FIPA-Propose",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "canal": canal,
                    "contexto_conversacion_id": contexto_id,
                    "nivel_indecision": nivel_indecision,
                    "cotizacion_actual": cotizacion_actual_num,
                    "servicio": servicio,
                    "cantidad_unidades": cantidad,
                    "complejidad": complejidad,
                    "fidelidad_cliente": fidelidad,
                    "descuento_solicitado_cliente": desc_solicitado,
                    "promocion_codigo": promocion["codigo"],
                },
            )
            return

        asentimiento_con_cotizacion = (
            tiene_cotizacion
            and not cotizacion_aprobada_actual
            and _es_respuesta_afirmativa(texto)
        )
        if (
            tipo_evento == "aceptar_cotizacion"
            or tipo_evento == "solicitud_agendamiento"
            or estado_reserva.get("paso") == "esperando_horario"
            or analisis["intencion"] == "solicitud_agendamiento"
            or _es_aceptacion_explicita_cotizacion(texto)
            or asentimiento_con_cotizacion
        ):
            if (
                _es_aceptacion_explicita_cotizacion(texto)
                or asentimiento_con_cotizacion
                or tipo_evento == "aceptar_cotizacion"
            ):
                self.entorno.perfilesClientes.aprobar_ultima_cotizacion(
                    id_cliente, contexto_id
                )
                cotizacion_aprobada_actual = self.entorno.perfilesClientes.cotizacion_aprobada(
                    id_cliente, contexto_id
                )

            parametros_cliente = self.entorno.perfilesClientes.obtener_parametros_cotizacion(
                id_cliente, contexto_id
            )
            fecha_hora_actual = str(
                parametros_extraidos.get("fecha_hora", "")
            ).strip()
            fecha_hora_pendiente = str(
                parametros_cliente.get("fecha_agendamiento_pendiente")
                or parametros_cliente.get("fecha_hora")
                or ""
            ).strip()
            fecha_parcial = str(
                parametros_extraidos.get("fecha_agendamiento_fecha")
                or parametros_cliente.get("fecha_agendamiento_fecha")
                or ""
            ).strip()
            if fecha_hora_actual:
                fecha_hora = fecha_hora_actual
                self.entorno.perfilesClientes.establecer_fecha_agendamiento_pendiente(
                    id_cliente, fecha_hora, contexto_id
                )
            else:
                fecha_hora = fecha_hora_pendiente
            if fecha_parcial and not fecha_hora_actual:
                self.entorno.perfilesClientes.actualizar_parametros_cotizacion(
                    id_cliente,
                    {
                        "fecha_agendamiento_fecha": fecha_parcial,
                        "fecha_hora_falta": True,
                    },
                    contexto_id,
                )

            if fecha_hora and _fecha_hora_ya_paso(fecha_hora):
                self.entorno.perfilesClientes.establecer_fecha_agendamiento_pendiente(
                    id_cliente, None, contexto_id
                )
                fecha_hora = ""
                mensaje = (
                    "La fecha indicada ya pasó. ¿Qué día y hora futuros te gustaría "
                    "reservar?"
                )
                res = self.responderLenguajeNatural(
                    "fecha_agendamiento_pasada", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {"resultado": "fecha_agendamiento_pasada", "envio": res}
                )
                return

            if not tiene_cotizacion:
                parametros_cotizacion = {
                    **parametros_cliente,
                    **{
                        clave: valor
                        for clave, valor in parametros_extraidos.items()
                        if clave
                        in {
                            "tipo_servicio",
                            "cantidad_unidades",
                            "horas_estimadas",
                            "sesiones_requeridas",
                            "cantidad_productos",
                            "complejidad",
                            "urgencia_dias",
                            "fecha_agendamiento_fecha",
                        }
                    },
                }
                if parametros_cotizacion:
                    self.entorno.perfilesClientes.actualizar_parametros_cotizacion(
                        id_cliente, parametros_cotizacion, contexto_id
                    )
                mensaje = (
                    "Para agendar primero necesitamos preparar una cotización. "
                    "Cuéntame qué servicio necesitas y te ayudamos a cotizarlo."
                )
                res = self.responderLenguajeNatural(
                    "cotizacion_requerida", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {"resultado": "cotizacion_requerida", "envio": res}
                )
                return

            if not cotizacion_aprobada_actual:
                mensaje = (
                    "Antes de agendar, confirma que aceptas la cotización enviada. "
                    "Puedes responder: «Acepto la cotización»."
                )
                res = self.responderLenguajeNatural(
                    "cotizacion_aprobacion_requerida", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {
                        "resultado": "cotizacion_aprobacion_requerida",
                        "envio": res,
                    }
                )
                return

            if not fecha_hora:
                self.entorno.perfilesClientes.actualizar_estado_conversacion(
                    id_cliente,
                    contexto_id,
                    {"estado_reserva": {**estado_reserva, "paso": "datos_cliente"}},
                )
                if not estado_reserva.get("nombre_cliente"):
                    estado_reserva["paso"] = "nombre_cliente"
                    mensaje = (
                        "Perfecto. Para agendar necesito tu nombre completo. "
                        "¿Cuál es tu nombre y apellido?"
                    )
                elif not estado_reserva.get("cedula"):
                    estado_reserva["paso"] = "cedula"
                    mensaje = "¿Cuál es tu número de identificación?"
                elif not estado_reserva.get("telefono"):
                    estado_reserva["paso"] = "telefono"
                    mensaje = "¿Cuál es tu teléfono de contacto?"
                else:
                    estado_reserva["paso"] = "esperando_horario"
                    mensaje = (
                        f"Ya tengo el día {fecha_parcial}. ¿A qué hora te gustaría la cita?"
                        if fecha_parcial
                        else "¿Qué día y hora te gustaría reservar?"
                    )
                self.entorno.perfilesClientes.actualizar_estado_conversacion(
                    id_cliente,
                    contexto_id,
                    {"estado_reserva": estado_reserva},
                )
                res = self.responderLenguajeNatural(
                    "datos_reserva_requeridos", canal, id_cliente, mensaje
                )
                self.marcar_ciclo_completado(
                    {
                        "resultado": "datos_reserva_pendientes",
                        "envio": res,
                    }
                )
                return

            try:
                datos_cliente_reserva = validar_datos_cliente_reserva(
                    estado_reserva.get("nombre_cliente"),
                    estado_reserva.get("cedula"),
                    estado_reserva.get("telefono"),
                )
            except ValueError:
                for paso, clave, mensaje in (
                    (
                        "nombre_cliente",
                        "nombre_cliente",
                        "Para agendar necesito tu nombre completo. ¿Cuál es tu nombre y apellido?",
                    ),
                    ("cedula", "cedula", "¿Cuál es tu número de identificación?"),
                    ("telefono", "telefono", "¿Cuál es tu teléfono de contacto?"),
                ):
                    if not estado_reserva.get(clave):
                        estado_reserva["paso"] = paso
                        self.entorno.perfilesClientes.actualizar_estado_conversacion(
                            id_cliente,
                            contexto_id,
                            {"estado_reserva": estado_reserva},
                        )
                        res = self.responderLenguajeNatural(
                            "datos_reserva_requeridos", canal, id_cliente, mensaje
                        )
                        self.marcar_ciclo_completado(
                            {
                                "resultado": "datos_reserva_pendientes",
                                "envio": res,
                            }
                        )
                        return
                estado_reserva["paso"] = "nombre_cliente"
                self.entorno.perfilesClientes.actualizar_estado_conversacion(
                    id_cliente,
                    contexto_id,
                    {"estado_reserva": estado_reserva},
                )
                res = self.responderLenguajeNatural(
                    "datos_reserva_requeridos",
                    canal,
                    id_cliente,
                    "Para agendar necesito verificar tus datos completos. ¿Cuál es tu nombre y apellido?",
                )
                self.marcar_ciclo_completado(
                    {"resultado": "datos_reserva_pendientes", "envio": res}
                )
                return

            estado_reserva.update(datos_cliente_reserva)
            estado_reserva["paso"] = "esperando_horario"
            self.entorno.perfilesClientes.actualizar_estado_conversacion(
                id_cliente, contexto_id, {"estado_reserva": estado_reserva}
            )

            parametros_cliente = self.entorno.perfilesClientes.obtener_parametros_cotizacion(
                id_cliente, contexto_id
            )
            servicio = (
                parametros_cliente.get("tipo_servicio")
                or parametros_extraidos.get("tipo_servicio")
            )
            rechazadas = estado_reserva.get("alternativas_rechazadas", [])
            await self.escalarCasoACoordinador(
                comportamiento=comportamiento,
                performativa="REQUEST",
                predicado="(solicitar_agendamiento fecha_hora cliente)",
                protocolo="FIPA-Query",
                conversation_id=conversation_id,
                payload_extra={
                    "id_cliente": id_cliente,
                    "nombre_cliente": datos_cliente_reserva["nombre_cliente"],
                    "canal": canal,
                    "fecha_hora": fecha_hora,
                    "servicio": servicio,
                    "cotizacion_aprobada": True,
                    "contexto_conversacion_id": contexto_id,
                    "alternativas_rechazadas": rechazadas,
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
            if desglose.get("es_personalizado"):
                cant_fotos = int(desglose.get("cantidad_fotos", 1))
                cant_personas = int(desglose.get("cantidad_personas", 1))
                locacion_txt = (
                    "en locación externa"
                    if desglose.get("locacion") == "exterior"
                    else "en nuestro estudio"
                )
                fotos_txt = "1 foto editada en alta resolución" if cant_fotos == 1 else f"{cant_fotos} fotos editadas en alta resolución"
                personas_txt = f" para {cant_personas} personas" if cant_personas > 1 else ""
                nota_paquete = ""
                if desglose.get("conveniencia_paquete_completo"):
                    fotos_ref = int(desglose.get("fotos_paquete_referencia", cant_fotos))
                    nota_paquete = (
                        f" Por ese mismo valor puedes llevar el paquete completo de {fotos_ref} fotos editadas."
                    )
                texto_cot = (
                    f"Para tu {desglose.get('servicio', 'servicio fotográfico a medida')}{personas_txt} "
                    f"({fotos_txt}, {locacion_txt}, con alistamiento de equipo, iluminación y retoque profesional), "
                    f"la cotización formal es de ${valor_total:,.0f} {moneda}.{nota_paquete} "
                    "¿Te gustaría agendar una cita para esta sesión o ajustar la cantidad de fotos o modalidad (estudio/exterior)?"
                )
            elif normalizar_unidad_comercial(
                desglose.get("unidad_cantidad", "")
            ) == "sesion":
                cantidad = int(desglose.get("cantidad_unidades", 0))
                tarifa_unitaria = float(desglose["tarifa_unitaria"])
                complejidad = desglose.get("complejidad")
                cantidad_texto = "sesión" if cantidad == 1 else "sesiones"
                complejidad_texto = (
                    str(complejidad) if complejidad is not None else "no aplica"
                )
                texto_cot = (
                    f"Tu cotización para '{desglose.get('servicio', 'Servicio')}' es de "
                    f"${valor_total:,.0f} {moneda}: {cantidad} {cantidad_texto} × "
                    f"${tarifa_unitaria:,.0f} {moneda}. Complejidad registrada: "
                    f"{complejidad_texto}; no se aplica un recargo porque no hay una "
                    "regla de precio por complejidad configurada. "
                    "¿Te gustaría agendar una cita para este servicio?"
                )
            else:
                costo_base = float(
                    desglose.get(
                        "costo_base", desglose.get("costo_ingenieria_base_cop", 0)
                    )
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
                    f"Ajuste por alcance/urgencia: ${ajuste:,.0f} {moneda}). "
                    "¿Te gustaría agendar una cita para iniciar?"
                )
            hechos_cot = (
                [f"${valor_total:,.0f}"]
                if desglose.get("es_personalizado")
                else [
                    str(desglose.get("servicio", "")),
                    f"${valor_total:,.0f}",
                ]
            )
            res = self.responderLenguajeNatural(
                "cotizacion_formal",
                canal,
                id_cliente,
                texto_cot,
                hechos_obligatorios=hechos_cot,
            )
            self.marcar_ciclo_completado(
                {"resultado": "cotizacion_entregada", "valor_total": valor_total, "desglose": desglose, "envio": res}
            )

        elif perf == "REQUEST" and "solicitar_parametros_faltantes" in contenido:
            etiqueta_default = str(
                self.entorno.config_negocio.get("etiqueta_cantidad", "cantidad_unidades")
            )
            unidad = str(self.entorno.config_negocio.get("unidad_cantidad", "unidades"))
            faltantes = payload.get("faltantes", [etiqueta_default])
            descripciones = [
                self._describir_parametro_faltante(str(c), unidad) for c in faltantes
            ]
            texto_aclaracion = (
                "Con gusto te preparo la cotización. Para darte un valor exacto, "
                f"cuéntame por favor {', '.join(descripciones)}."
            )
            res = self.responderLenguajeNatural("pregunta_aclaratoria", canal, id_cliente, texto_aclaracion)
            self.marcar_ciclo_completado(
                {"resultado": "parametros_incompletos", "faltantes": faltantes, "envio": res}
            )

        elif perf == "INFORM" and "transferir_promocion" in contenido:
            try:
                descuento = float(payload.get("descuento_porciento"))
                valor_final = float(payload.get("valor_con_descuento"))
            except (TypeError, ValueError):
                descuento = 0.0
                valor_final = 0.0
            res = self.presentarPromocionAlCliente(
                descuento_porciento=descuento,
                canal=canal,
                id_cliente=id_cliente,
                valor_final=valor_final,
                conversation_id=(
                    str(payload["contexto_conversacion_id"])
                    if payload.get("contexto_conversacion_id")
                    else None
                ),
                codigo_promocion=(
                    str(payload["promocion_codigo"])
                    if payload.get("promocion_codigo")
                    else None
                ),
            )
            self.marcar_ciclo_completado(
                {
                    "resultado": "promocion_presentada"
                    if self.creencias.get("promocionAprobada") == descuento
                    else "tarifa_base_mantenida",
                    "descuento_porciento": descuento,
                    "valor_final": valor_final,
                    "envio": res,
                }
            )

        elif perf == "REFUSE" and "descuento_no_viable" in contenido:
            contexto_id = (
                str(payload["contexto_conversacion_id"])
                if payload.get("contexto_conversacion_id")
                else None
            )
            contexto = self.entorno.perfilesClientes.obtener_estado_conversacion(
                id_cliente, contexto_id
            )
            perfil = self.entorno.perfilesClientes.obtener_o_crear_perfil(id_cliente)
            valor_base = (
                contexto.get("ultima_cotizacion")
                if contexto_id
                else perfil.get("ultima_cotizacion")
            )
            moneda = str(self.entorno.config_negocio.get("moneda", "COP"))
            if valor_base is not None:
                texto_refuse = (
                    f"La tarifa formal vigente es de ${float(valor_base):,.0f} "
                    f"{moneda}. No hay una promoción autorizada aplicable, "
                    "así que mantenemos ese precio."
                )
            else:
                texto_refuse = (
                    "Para revisar un precio o incentivo primero necesitamos preparar "
                    "una cotización formal del servicio."
                )
            res = self.responderLenguajeNatural("mantener_tarifa_base", canal, id_cliente, texto_refuse)
            self.marcar_ciclo_completado(
                {"resultado": "descuento_rechazado", "envio": res}
            )

        elif perf in {"FAILURE", "REFUSE"} and "error_agendamiento" in contenido:
            if payload.get("estado") == "pendiente":
                texto_error = (
                    "Recibimos tu solicitud de cita, pero no pudimos verificar la disponibilidad "
                    "del calendario externo. Quedó pendiente y todavía no está confirmada; "
                    "te contactaremos cuando podamos validarla."
                )
                resultado = "agendamiento_pendiente"
            else:
                texto_error = (
                    "No pudimos confirmar la cita porque no se completó su registro. "
                    "Por favor, intenta nuevamente o contacta directamente al negocio."
                )
                resultado = "error_agendamiento"
            res = self.responderLenguajeNatural(
                resultado, canal, id_cliente, texto_error
            )
            self.marcar_ciclo_completado(
                {
                    "resultado": resultado,
                    "agendamiento_id": payload.get("agendamiento_id"),
                    "estado": payload.get("estado"),
                    "google_calendar_estado": payload.get(
                        "google_calendar_estado", "error"
                    ),
                    "envio": res,
                }
            )

        elif perf == "INFORM" and "reserva_confirmada" in contenido:
            fecha_hora = str(payload.get("fecha_hora", ""))
            google_estado = str(payload.get("google_calendar_estado", "error"))
            contexto_id = (
                str(payload["contexto_conversacion_id"])
                if payload.get("contexto_conversacion_id")
                else None
            )
            self.entorno.perfilesClientes.establecer_fecha_agendamiento_pendiente(
                id_cliente, None, contexto_id
            )
            self.entorno.perfilesClientes.actualizar_estado_conversacion(
                id_cliente,
                contexto_id,
                {"estado_reserva": {}},
            )
            if google_estado == "sincronizado":
                texto_cita = (
                    f"Cita guardada exitosamente y confirmada para el {fecha_hora}. "
                    "También quedó sincronizada con Google Calendar."
                )
            elif google_estado == "error":
                texto_cita = (
                    f"Cita guardada exitosamente y confirmada para el {fecha_hora} "
                    "en la agenda del negocio. "
                    "No se pudo sincronizar con Google Calendar; la reserva sí quedó "
                    "persistida en la base de datos."
                )
            elif google_estado == "no_configurado":
                texto_cita = (
                    f"Cita guardada exitosamente y confirmada para el {fecha_hora} "
                    "en la agenda del negocio. "
                    "Google Calendar no está configurado."
                )
            else:
                texto_cita = (
                    f"Cita guardada exitosamente y confirmada para el {fecha_hora} "
                    "en la agenda del negocio. "
                    f"Estado de Google Calendar: {google_estado}."
                )
            res = self.responderLenguajeNatural(
                "confirmacion_cita",
                canal,
                id_cliente,
                texto_cita,
                hechos_obligatorios=[fecha_hora] if fecha_hora else None,
            )
            self.marcar_ciclo_completado(
                {
                    "resultado": "cita_confirmada",
                    "fecha_hora": fecha_hora,
                    "agendamiento_id": payload.get("agendamiento_id"),
                    "registrado_en": payload.get("registrado_en"),
                    "google_calendar_estado": google_estado,
                    "google_event_id": payload.get("google_event_id"),
                    "google_event_link": payload.get("google_event_link"),
                    "envio": res,
                }
            )

        elif perf == "INFORM" and "ofrecer_alternativas" in contenido:
            alternativas = payload.get("alternativas", [])
            contexto_id = (
                str(payload["contexto_conversacion_id"])
                if payload.get("contexto_conversacion_id")
                else None
            )
            contexto = self.entorno.perfilesClientes.obtener_estado_conversacion(
                id_cliente, contexto_id
            )
            estado_reserva = dict(contexto.get("estado_reserva", {}))
            rechazadas = list(estado_reserva.get("alternativas_rechazadas", []))
            horario_intentado = str(payload.get("fecha_hora", "")).strip()
            if horario_intentado and horario_intentado not in rechazadas:
                rechazadas.append(horario_intentado)
            estado_reserva.update(
                {
                    "paso": "esperando_horario",
                    "alternativas_rechazadas": rechazadas,
                    "alternativas_ofrecidas": list(alternativas),
                }
            )
            self.entorno.perfilesClientes.actualizar_estado_conversacion(
                id_cliente,
                contexto_id,
                {"estado_reserva": estado_reserva},
            )
            texto_alt = (
                "El horario solicitado ya se encuentra ocupado en nuestra agenda. "
                f"Te propongo estos bloques disponibles: {', '.join(alternativas)}."
                if alternativas
                else "El horario solicitado está ocupado y no pude verificar alternativas disponibles. "
                "¿Qué otro día u hora te gustaría consultar?"
            )
            res = self.responderLenguajeNatural(
                "sugerir_horarios_alternativos",
                canal,
                id_cliente,
                texto_alt,
                hechos_obligatorios=list(alternativas) if alternativas else None,
            )
            self.marcar_ciclo_completado(
                {"resultado": "horarios_alternativos", "alternativas": alternativas, "envio": res}
            )

        elif perf == "REFUSE" and "requiere_cotizacion_aprobada" in contenido:
            texto_req = (
                "Antes de agendar una cita, necesitamos contar con una cotización "
                "formal aprobada. Si ya la recibiste, responde «Acepto la cotización»."
            )
            res = self.responderLenguajeNatural(
                "cotizacion_aprobacion_requerida", canal, id_cliente, texto_req
            )
            self.marcar_ciclo_completado(
                {"resultado": "cotizacion_aprobacion_requerida", "envio": res}
            )

        elif perf == "REFUSE" and "fuera_de_alcance_operativo" in contenido:
            texto_no_viable = (
                "En este momento esa solicitud se encuentra fuera de nuestro alcance operativo. "
                "Si deseas consultar otros servicios de nuestro catálogo, con gusto te ayudo."
            )
            res = self.responderLenguajeNatural(
                "solicitud_no_viable_cierre", canal, id_cliente, texto_no_viable
            )
            self.marcar_ciclo_completado(
                {
                    "resultado": "solicitud_no_viable",
                    "motivo": payload.get("motivo", "fuera_de_alcance_operativo"),
                    "envio": res,
                }
            )

        elif perf == "INFORM" and "confirmar_cierre_conversacion" in contenido:
            texto_despedida = (
                "Entendido, registramos el cierre de tu solicitud. "
                "Muchas gracias por tu tiempo y quedamos atentos si más adelante necesitas nuestros servicios."
            )
            res = self.responderLenguajeNatural(
                "despedida_cordial_y_registro", canal, id_cliente, texto_despedida
            )
            self.marcar_ciclo_completado(
                {"resultado": "oportunidad_descartada", "envio": res}
            )

        elif perf == "CANCEL" and "cancelar_accion" in contenido:
            self.registrar_accion("pausarChatAutomatico()")
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

