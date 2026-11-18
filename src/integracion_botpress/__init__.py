"""
Paquete de integracion con Botpress (WhatsApp, Webchat, Messenger e Instagram)
y panel web Monitor FIPA-ACL + Human-in-the-Loop.
"""

from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.integracion_botpress.servidor_botpress import crear_app_botpress

__all__ = ["ConfiguracionBotpress", "crear_app_botpress"]
