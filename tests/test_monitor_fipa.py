from __future__ import annotations

import asyncio
import io
import json
import re
from typing import Any

import pytest
from rich.console import Console

from src.comunicacion.fipa_acl import crear_mensaje_fipa
from src.comunicacion.monitor_fipa import MonitorFIPAACL
from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial
from src.integracion_botpress.configuracion import ConfiguracionBotpress
from src.persistencia.repositorio import RepositorioSQLite


def _mensaje_fipa(conversation_id: str = "bp-whatsapp-573001112233-1"):
    return crear_mensaje_fipa(
        sender_jid="ventas@localhost",
        receiver_jid="coordinador@localhost",
        performative="REQUEST",
        predicado_contenido=(
            "(escalar_caso cliente 573001112233 Maria Lopez)"
        ),
        protocol="FIPA-Request",
        conversation_id=conversation_id,
        payload_extra={
            "telefono": "573001112233",
            "nombre": "Maria Lopez",
            "credencial": "no-debe-aparecer",
        },
    )


def test_monitor_imprime_linea_compacta_y_redacta_datos_sensibles() -> None:
    salida = io.StringIO()
    consola = Console(
        file=salida,
        width=240,
        force_terminal=False,
        no_color=True,
        markup=False,
    )
    monitor = MonitorFIPAACL(consola=consola)

    registro = monitor.registrar_mensaje_fipa(_mensaje_fipa())
    panel = salida.getvalue()
    fipa = registro["fipa_json"]

    assert panel.count("FIPA-ACL Monitor") == 1
    assert "[FIPA]" not in panel
    assert re.search(
        r"\[FIPA-ACL Monitor \| FIPA-Request\]",
        panel,
    )
    assert "AgenteVentas -> AgenteCoordinador" in panel
    assert "REQUEST" in panel
    assert "<ACL:REQUEST>" in panel
    assert '"content": "accion=escalar_caso"' in panel
    assert '"conversation-id": "conv-' in panel
    assert re.search(r'"timestamp": "\d{4}-\d{2}-\d{2}T[^"]+-05:00"', panel)
    assert "send_ok_local" in panel
    assert "recepción no confirmada" in panel
    assert "573001112233" not in panel
    assert "Maria Lopez" not in panel
    assert "no-debe-aparecer" not in panel

    assert fipa[":sender"] == "AgenteVentas"
    assert fipa[":receiver"] == "AgenteCoordinador"
    assert fipa[":performative"] == "REQUEST"
    assert fipa[":protocol"] == "FIPA-Request"
    assert fipa[":content"] == "accion=escalar_caso"
    assert fipa[":conversation-id"].startswith("conv-")
    assert "573001112233" not in json.dumps(registro)
    assert "Maria Lopez" not in json.dumps(registro)
    assert "no-debe-aparecer" not in json.dumps(registro)
    assert registro["resultado_envio"] == "send_ok_local"
    assert fipa[":timestamp"].endswith("-05:00")


@pytest.mark.parametrize(
    "performativa",
    ["REQUEST", "AGREE", "INFORM", "CONFIRM", "REFUSE", "FAILURE"],
)
def test_monitor_rich_registra_performativas_sin_exponer_payload(
    performativa: str,
) -> None:
    salida = io.StringIO()
    monitor = MonitorFIPAACL(
        consola=Console(file=salida, force_terminal=False, no_color=True)
    )
    mensaje = crear_mensaje_fipa(
        sender_jid="coordinador@localhost",
        receiver_jid="cotizaciones@localhost",
        performative=performativa,
        predicado_contenido="(parametros_cliente telefono 573001112233)",
        protocol="FIPA-Request",
        conversation_id="prueba-sensible",
    )

    monitor.registrar_mensaje_fipa(mensaje)

    salida_panel = salida.getvalue()
    assert performativa in salida_panel
    assert "<ACL:" + performativa + ">" in salida_panel
    assert "573001112233" not in salida_panel
    assert len(monitor.historial_mensajes) == 1


def test_monitor_silencioso_conserva_historial_sin_imprimir() -> None:
    salida = io.StringIO()
    monitor = MonitorFIPAACL(
        consola=Console(file=salida, force_terminal=False, no_color=True),
        silencioso=True,
    )

    monitor.registrar_mensaje_fipa(_mensaje_fipa())

    assert salida.getvalue() == ""
    assert len(monitor.historial_mensajes) == 1


def test_traza_de_entrada_no_imprime_identidad_ni_texto_del_cliente() -> None:
    salida = io.StringIO()
    consola = Console(file=salida, force_terminal=False, no_color=True)
    servicio = ServicioMultiagenteComercial(
        config_botpress=ConfiguracionBotpress(),
        repositorio=RepositorioSQLite(":memory:"),
        usar_xmpp_embebido=False,
        consola=consola,
        silencioso=False,
    )

    servicio.entorno.eventoLlegadaMensaje(
        "consulta_servicio",
        {
            "id_cliente": "573001112233",
            "nombre": "Nombre Privado",
            "canal": "WhatsApp",
            "texto": "Mi correo privado@example.com y mi teléfono 573001112233",
        },
    )

    texto_consola = salida.getvalue()
    assert "[ENTORNO]" in texto_consola
    assert "mensaje_recibido" in texto_consola
    assert "573001112233" not in texto_consola
    assert "Nombre Privado" not in texto_consola
    assert "privado@example.com" not in texto_consola


def test_enviar_fipa_registra_solo_despues_de_send_exitoso() -> None:
    servicio = ServicioMultiagenteComercial(
        config_botpress=ConfiguracionBotpress(),
        repositorio=RepositorioSQLite(":memory:"),
        usar_xmpp_embebido=False,
        silencioso=True,
    )
    monitor = servicio.monitor

    class ComportamientoFalso:
        def __init__(self) -> None:
            self.mensajes: list[Any] = []

        async def send(self, mensaje: Any) -> None:
            assert not monitor.historial_mensajes
            self.mensajes.append(mensaje)

    class ComportamientoFallido:
        async def send(self, _mensaje: Any) -> None:
            raise RuntimeError("fallo local de envio")

    async def _run() -> None:
        comportamiento = ComportamientoFalso()
        await servicio.ventas.enviar_fipa(
            comportamiento=comportamiento,
            receptor_jid="coordinador@localhost",
            performativa="REQUEST",
            predicado_contenido="(escalar_caso cotizacion)",
            protocolo="FIPA-Request",
            conversation_id="bp-whatsapp-573001112233-1",
        )
        assert len(comportamiento.mensajes) == 1
        assert len(monitor.historial_mensajes) == 1
        assert monitor.historial_mensajes[0]["resultado_envio"] == "send_ok_local"

        with pytest.raises(RuntimeError, match="fallo local de envio"):
            await servicio.ventas.enviar_fipa(
                comportamiento=ComportamientoFallido(),
                receptor_jid="coordinador@localhost",
                performativa="REQUEST",
                predicado_contenido="(escalar_caso cotizacion)",
                protocolo="FIPA-Request",
                conversation_id="bp-whatsapp-573001112233-2",
            )
        assert len(monitor.historial_mensajes) == 1

    asyncio.run(_run())
