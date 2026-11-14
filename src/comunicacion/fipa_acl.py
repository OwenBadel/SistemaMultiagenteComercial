"""
Estandar de mensajeria FIPA-ACL sobre `spade.message.Message`.
Configura explicitamente en metadata: performative, ontology="Comercial-SMA",
protocol ("FIPA-Request", "FIPA-Propose", "FIPA-Query") y stereotype.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from spade.message import Message

ONTOLOGIA_COMERCIAL = "Comercial-SMA"

PROTOCOLOS_VALIDOS = {
    "FIPA-Request",
    "FIPA-Propose",
    "FIPA-Query",
}

PERFORMATIVAS_SOPORTADAS = {
    "REQUEST",
    "AGREE",
    "INFORM",
    "FAILURE",
    "PROPOSE",
    "ACCEPT-PROPOSAL",
    "REJECT-PROPOSAL",
    "REFUSE",
    "QUERY-IF",
    "CONFIRM",
    "CANCEL",
}


def crear_mensaje_fipa(
    sender_jid: str,
    receiver_jid: str,
    performative: str,
    predicado_contenido: str,
    protocol: str = "FIPA-Request",
    conversation_id: Optional[str] = None,
    payload_extra: Optional[Dict[str, Any]] = None,
) -> Message:
    """
    Construye una instancia oficial de `spade.message.Message` cumpliendo con
    los metadatos FIPA-ACL exigidos por el proyecto.
    """
    performativa_norm = performative.strip().upper()
    if performativa_norm not in PERFORMATIVAS_SOPORTADAS:
        raise ValueError(f"Performativa FIPA-ACL no soportada: {performativa_norm}")
    if protocol not in PROTOCOLOS_VALIDOS:
        raise ValueError(f"Protocolo FIPA no soportado: {protocol}")

    conv_id = conversation_id or f"conv-{uuid.uuid4().hex[:8]}"
    msg = Message(
        to=receiver_jid,
        sender=sender_jid,
        body=predicado_contenido,
        thread=conv_id,
    )
    msg.set_metadata("performative", performativa_norm)
    msg.set_metadata("ontology", ONTOLOGIA_COMERCIAL)
    msg.set_metadata("protocol", protocol)
    msg.set_metadata("stereotype", f"<<ACL:{performativa_norm}>>")
    msg.set_metadata("conversation_id", conv_id)
    if payload_extra is not None:
        msg.set_metadata(
            "payload_json", json.dumps(payload_extra, ensure_ascii=False)
        )
    return msg


def extraer_payload_fipa(msg: Message) -> Dict[str, Any]:
    """Deserializa el payload estructurado embebido en los metadatos FIPA-ACL."""
    raw = msg.get_metadata("payload_json")
    if not raw:
        return {}
    return dict(json.loads(raw))


def formatear_nombre_agente(jid_o_nombre: str) -> str:
    """Normaliza un JID SPADE (ej. `ventas@localhost`) al nombre de clase MAS-ML."""
    base = str(jid_o_nombre).split("@")[0].split("/")[0].lower()
    mapa = {
        "coordinador": "AgenteCoordinador",
        "agentecoordinador": "AgenteCoordinador",
        "ventas": "AgenteVentas",
        "agenteventas": "AgenteVentas",
        "cotizaciones": "AgenteCotizaciones",
        "agentecotizaciones": "AgenteCotizaciones",
        "promociones": "AgentePromociones",
        "agentepromociones": "AgentePromociones",
        "agendamientos": "AgenteAgendamientos",
        "agenteagendamientos": "AgenteAgendamientos",
    }
    return mapa.get(base, str(jid_o_nombre))


def formatear_etiqueta_diapositiva(msg: Message) -> str:
    """
    Genera exactamente el bloque de etiqueta FIPA-ACL exigido en los
    diagramas de secuencia del curso:
    [AgenteEmisor -> AgenteReceptor]
    PERFORMATIVA
    (predicado_contenido parametros)
    <<ACL:PERFORMATIVA>>
    """
    emisor = formatear_nombre_agente(str(msg.sender or ""))
    receptor = formatear_nombre_agente(str(msg.to or ""))
    perf = (msg.get_metadata("performative") or "INFORM").upper()
    contenido = msg.body or ""
    estereotipo = msg.get_metadata("stereotype") or f"<<ACL:{perf}>>"
    return f"[{emisor} -> {receptor}]\n{perf}\n{contenido}\n{estereotipo}"


def serializar_estructura_fipa_json(msg: Message, timestamp: str) -> Dict[str, Any]:
    """Serializa el mensaje FIPA-ACL con claves formales FIPA (:sender, :receiver, ...)."""
    perf = (msg.get_metadata("performative") or "INFORM").upper()
    return {
        ":performative": perf,
        ":sender": str(msg.sender),
        ":receiver": str(msg.to),
        ":content": msg.body,
        ":ontology": msg.get_metadata("ontology") or ONTOLOGIA_COMERCIAL,
        ":protocol": msg.get_metadata("protocol") or "FIPA-Request",
        ":conversation-id": msg.get_metadata("conversation_id") or msg.thread,
        ":timestamp": timestamp,
    }

