"""
Cliente asincrono para Meta Graph API (WhatsApp Cloud API, Facebook Messenger
e Instagram Direct) usando httpx.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from src.integracion_meta.configuracion import ConfiguracionMeta


class ClienteMetaGraphAPI:
    """
    Cliente HTTP encargado de entregar los mensajes generados por los agentes
    hacia los canales oficiales de Meta Platforms.
    """

    def __init__(
        self,
        config: ConfiguracionMeta,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config
        self._base_url = f"https://graph.facebook.com/{self.config.meta_api_version}"
        self._client_externo = http_client

    async def _post_json(
        self, url: str, token: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if self._client_externo is not None:
            resp = await self._client_externo.post(
                url, headers=headers, json=payload, timeout=15.0
            )
            return self._procesar_respuesta_http(resp)

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            return self._procesar_respuesta_http(resp)

    @staticmethod
    def _procesar_respuesta_http(resp: httpx.Response) -> Dict[str, Any]:
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        if resp.status_code >= 400:
            return {
                "enviado": False,
                "estado": "error_meta_api",
                "status_code": resp.status_code,
                "detalle": data,
            }
        return {
            "enviado": True,
            "estado": "enviado_meta",
            "status_code": resp.status_code,
            "detalle": data,
        }

    async def enviar_mensaje_whatsapp(
        self, telefono_destino: str, texto: str
    ) -> Dict[str, Any]:
        """Envia un mensaje de texto via WhatsApp Cloud API."""
        if not (
            self.config.whatsapp_access_token and self.config.whatsapp_phone_number_id
        ):
            return {
                "enviado": False,
                "estado": "pendiente_token_meta",
                "canal": "WhatsApp",
                "destino": telefono_destino,
                "mensaje": texto,
            }

        url = f"{self._base_url}/{self.config.whatsapp_phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": telefono_destino,
            "type": "text",
            "text": {"preview_url": False, "body": texto},
        }
        resultado = await self._post_json(
            url, self.config.whatsapp_access_token, payload
        )
        return {"canal": "WhatsApp", "destino": telefono_destino, **resultado}

    async def enviar_mensaje_messenger(
        self, psid_destino: str, texto: str
    ) -> Dict[str, Any]:
        """Envia un mensaje de texto via Facebook Messenger Platform."""
        if not self.config.messenger_page_access_token:
            return {
                "enviado": False,
                "estado": "pendiente_token_meta",
                "canal": "Facebook",
                "destino": psid_destino,
                "mensaje": texto,
            }

        url = f"{self._base_url}/me/messages"
        payload = {
            "recipient": {"id": psid_destino},
            "messaging_type": "RESPONSE",
            "message": {"text": texto},
        }
        resultado = await self._post_json(
            url, self.config.messenger_page_access_token, payload
        )
        return {"canal": "Facebook", "destino": psid_destino, **resultado}

    async def enviar_mensaje_instagram(
        self, igsid_destino: str, texto: str
    ) -> Dict[str, Any]:
        """Envia un mensaje de texto directo via Instagram Messaging API."""
        token = self.config.instagram_access_token or self.config.messenger_page_access_token
        if not token:
            return {
                "enviado": False,
                "estado": "pendiente_token_meta",
                "canal": "Instagram",
                "destino": igsid_destino,
                "mensaje": texto,
            }

        cuenta_id = self.config.instagram_account_id or "me"
        url = f"{self._base_url}/{cuenta_id}/messages"
        payload = {
            "recipient": {"id": igsid_destino},
            "message": {"text": texto},
        }
        resultado = await self._post_json(url, token, payload)
        return {"canal": "Instagram", "destino": igsid_destino, **resultado}

    async def enviar_mensaje_canal(
        self, canal: str, id_cliente: str, texto: str
    ) -> Dict[str, Any]:
        """Enruta la respuesta al canal Meta correspondiente (WhatsApp, Facebook o Instagram)."""
        canal_norm = canal.strip().lower()
        if canal_norm == "whatsapp":
            return await self.enviar_mensaje_whatsapp(id_cliente, texto)
        if canal_norm in {"facebook", "messenger"}:
            return await self.enviar_mensaje_messenger(id_cliente, texto)
        if canal_norm == "instagram":
            return await self.enviar_mensaje_instagram(id_cliente, texto)
        return {
            "enviado": False,
            "estado": "canal_no_soportado",
            "canal": canal,
            "destino": id_cliente,
        }

    async def notificar_humano_whatsapp(
        self, motivo: str, id_cliente: str, canal: str
    ) -> Dict[str, Any]:
        """Notifica al profesional humano por WhatsApp cuando ocurre escalamiento critico."""
        if not self.config.admin_whatsapp_number:
            return {
                "enviado": False,
                "estado": "pendiente_admin_whatsapp",
                "canal": "WhatsApp",
            }
        aviso = (
            f"[ALERTA ESCALAMIENTO HUMANO]\n"
            f"Cliente: {id_cliente} ({canal})\n"
            f"Motivo: {motivo}\n"
            f"La automatizacion del chat fue pausada para atencion directa."
        )
        return await self.enviar_mensaje_whatsapp(
            telefono_destino=self.config.admin_whatsapp_number, texto=aviso
        )

