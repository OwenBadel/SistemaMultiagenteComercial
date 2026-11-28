"""
Cliente directo para Meta WhatsApp Cloud API (Graph API v21.0).
Permite responder al cliente y notificar al WhatsApp personal del dueno sin Botpress,
manteniendo compatibilidad total cuando tambien se usa Botpress.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any, Dict, Optional

import httpx

from src.comercial.repositorio_multitenant import normalizar_telefono

_RE_PHONE_NUMBER_ID = re.compile(r"^\d{5,30}$")


def es_phone_number_id_valido(phone_number_id: str) -> bool:
    """Valida que el Phone Number ID de Meta sea estrictamente numerico."""
    return bool(_RE_PHONE_NUMBER_ID.fullmatch(str(phone_number_id or "").strip()))


def verificar_firma_meta(
    cuerpo_raw: bytes,
    firma_header: str,
    app_secret: str,
    permitir_sin_secreto: bool = False,
) -> bool:
    """Valida X-Hub-Signature-256 de Meta (fail-closed salvo modo desarrollo explicito)."""
    if not app_secret:
        return bool(permitir_sin_secreto)
    if not firma_header or not firma_header.startswith("sha256="):
        return False
    esperada = "sha256=" + hmac.new(
        app_secret.encode("utf-8"), cuerpo_raw, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(esperada, firma_header)


def enviar_mensaje_whatsapp_cloud(
    phone_number_id: str,
    access_token: str,
    destinatario: str,
    texto: str,
    http_client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Envia un mensaje de texto saliente mediante WhatsApp Cloud API."""
    pnid = str(phone_number_id or "").strip()
    if not es_phone_number_id_valido(pnid):
        return {"enviado": False, "motivo": "phone_number_id_invalido"}

    numero = normalizar_telefono(destinatario)
    if not access_token or not numero or not texto.strip():
        return {"enviado": False, "motivo": "credenciales_o_destino_incompleto"}

    url = f"https://graph.facebook.com/v21.0/{pnid}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": numero,
        "type": "text",
        "text": {"preview_url": True, "body": texto[:4096]},
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    try:
        if http_client is not None:
            resp = http_client.post(url, json=payload, headers=headers, timeout=12.0)
        else:
            resp = httpx.post(url, json=payload, headers=headers, timeout=12.0)
        resp.raise_for_status()
        return {"enviado": True, "respuesta": resp.json()}
    except Exception as exc:
        return {"enviado": False, "error": type(exc).__name__}
