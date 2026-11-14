"""
Runtime SPADE autocontenido compatible con:
1. Servidor XMPP embebido oficial de SPADE (`spade.run(..., embedded_xmpp_server=True)`).
2. Modo Contenedor Local SPADE en memoria (para pruebas rápidas con pytest o
   cuando se requiere reiniciar múltiples bucles asyncio en una misma sesión).

Autoría: Owen Badel Hooker
"""
from __future__ import annotations

import asyncio
from typing import Iterable

from spade.agent import Agent
from spade.behaviour import FSMBehaviour
from spade.container import Container


async def iniciar_agentes_spade(
    agentes: Iterable[Agent],
    usar_xmpp_embebido: bool = True,
) -> None:
    """
    Inicia los agentes SPADE:
    - Con `usar_xmpp_embebido=True`, ejecuta `await agente.start(auto_register=True)`
      contra el servidor PyJabber embebido de SPADE.
    - Con `usar_xmpp_embebido=False`, registra los agentes directamente en el
      `spade.container.Container` enlazado al event loop actual y arranca sus
      comportamientos asíncronos (`CyclicBehaviour` / `OneShotBehaviour`).
    """
    lista = list(agentes)
    if usar_xmpp_embebido:
        for agente in lista:
            await agente.start(auto_register=True)
        return

    contenedor = Container()
    contenedor.reset()
    loop_actual = asyncio.get_running_loop()
    contenedor.loop = loop_actual

    for agente in lista:
        contenedor.register(agente)
        agente.set_loop(loop_actual)
        await agente.setup()
        agente._alive.set()
        for comportamiento in agente.behaviours:
            if not comportamiento.is_running:
                comportamiento.set_agent(agente)
                if issubclass(type(comportamiento), FSMBehaviour):
                    for _, estado in comportamiento.get_states().items():
                        estado.set_agent(agente)
                comportamiento.start()


async def detener_agentes_spade(
    agentes: Iterable[Agent],
    usar_xmpp_embebido: bool = True,
) -> None:
    """Detiene limpiamente todos los agentes y sus comportamientos SPADE."""
    lista = list(agentes)
    for agente in lista:
        if usar_xmpp_embebido:
            await agente.stop()
        else:
            for comportamiento in agente.behaviours:
                comportamiento.kill()
            agente._alive.clear()
            Container().unregister(str(agente.jid))
    if not usar_xmpp_embebido:
        # Cede un tick para que las tareas canceladas terminen limpiamente
        await asyncio.sleep(0.02)
