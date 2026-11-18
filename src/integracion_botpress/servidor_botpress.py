"""
Servidor FastAPI para integracion directa con Botpress (puente hacia WhatsApp,
Webchat, Messenger e Instagram), Monitor FIPA-ACL en vivo y consola Human-in-the-Loop.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import hmac
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial


class PeticionMensajeBotpress(BaseModel):
    """
    Payload flexible aceptado desde Botpress Studio (Execute Code / HTTP Action).
    Soporta tanto nombres en espanol (`canal`, `id_cliente`, `nombre`, `texto`)
    como aliases comunes de Botpress (`channel`, `userId`, `conversationId`, `text`).
    """

    canal: Optional[str] = None
    channel: Optional[str] = None
    id_cliente: Optional[str] = None
    userId: Optional[str] = None
    conversationId: Optional[str] = None
    nombre: Optional[str] = None
    userName: Optional[str] = None
    texto: Optional[str] = None
    text: Optional[str] = None
    mensaje: Optional[str] = None
    tipo_evento_sugerido: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def normalizar(self) -> Dict[str, Any]:
        """Normaliza los campos entrantes hacia los parametros del SMA."""
        canal_final = (self.canal or self.channel or "Botpress").strip()
        id_final = (
            self.id_cliente
            or self.userId
            or self.conversationId
            or "botpress_cliente_001"
        ).strip()
        nombre_final = (
            self.nombre or self.userName or f"{canal_final} {id_final}"
        ).strip()
        texto_final = (self.texto or self.text or self.mensaje or "").strip()
        return {
            "canal": canal_final,
            "id_cliente": id_final,
            "nombre": nombre_final,
            "texto": texto_final,
        }


class PeticionRespuestaHumana(BaseModel):
    id_cliente: str
    texto: str
    canal: Optional[str] = None


class PeticionReactivarCliente(BaseModel):
    id_cliente: str


class PeticionCambioDisponibilidad(BaseModel):
    fecha_hora: str
    estado: str
    cliente: Optional[str] = None


def _validar_api_key_botpress(
    api_key_config: str,
    x_api_key: Optional[str],
    authorization: Optional[str],
) -> None:
    """Valida API Key opcional solo si BOTPRESS_API_KEY fue configurada en .env."""
    if not api_key_config:
        return
    token_recibido = x_api_key or ""
    if not token_recibido and authorization and authorization.lower().startswith("bearer "):
        token_recibido = authorization[7:].strip()
    if not token_recibido or not hmac.compare_digest(api_key_config, token_recibido):
        raise HTTPException(status_code=401, detail="API Key de Botpress invalida")


def crear_app_botpress(servicio_sma: ServicioMultiagenteComercial) -> FastAPI:
    """Construye la aplicacion FastAPI conectada a Botpress y al SMA en SPADE."""
    app = FastAPI(
        title="SistemaMultiagenteComercial - API Botpress y Monitor FIPA-ACL",
        version="2.0.0",
    )

    @app.get("/", response_class=HTMLResponse)
    async def panel_principal() -> str:
        return _HTML_PANEL_CONTROL

    @app.get("/health")
    async def healthcheck() -> Dict[str, Any]:
        return {
            "status": "ok",
            "servicio": "SistemaMultiagenteComercial-Botpress",
            "agentes_activos": sum(1 for a in servicio_sma.agentes if a.is_alive()),
        }

    @app.post("/api/botpress/mensaje")
    async def procesar_mensaje_botpress(
        req: PeticionMensajeBotpress,
        x_api_key: Optional[str] = Header(default=None),
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        """
        Endpoint oficial llamado por Botpress cuando llega un mensaje del cliente
        desde WhatsApp, Webchat, Facebook Messenger o Instagram.
        """
        _validar_api_key_botpress(
            servicio_sma.config_botpress.botpress_api_key,
            x_api_key,
            authorization,
        )
        datos = req.normalizar()
        if not datos["texto"]:
            raise HTTPException(
                status_code=400, detail="El campo 'texto' no puede estar vacio"
            )

        return await servicio_sma.procesar_mensaje_entrante(
            canal=datos["canal"],
            id_cliente=datos["id_cliente"],
            nombre_cliente=datos["nombre"],
            texto=datos["texto"],
            tipo_evento_sugerido=req.tipo_evento_sugerido,
            metadata_extra=req.metadata,
        )

    @app.get("/api/estado")
    async def obtener_estado_sistema() -> Dict[str, Any]:
        """Retorna el estado completo en vivo del SMA, FIPA-ACL y Human-in-the-Loop."""
        return {
            "canal_integracion": "Botpress API Bridge",
            "api_key_activa": bool(servicio_sma.config_botpress.botpress_api_key),
            "modelos_ml": {
                "random_forest_r2": round(servicio_sma.modelo_rf.r2_entrenamiento, 4),
                "knn_r2_cv": round(servicio_sma.modelo_knn.r2_entrenamiento, 4),
                "registros_tarifas": len(servicio_sma.df_tarifas),
                "registros_ofertas": len(servicio_sma.df_ofertas),
            },
            "agentes": [
                {
                    "nombre": ag.nombre_agente,
                    "rol": ag.nombre_rol,
                    "arquitectura": ag.arquitectura_tipo,
                    "vivo": ag.is_alive(),
                }
                for ag in servicio_sma.agentes
            ],
            "mensajes_fipa": servicio_sma.monitor.historial_mensajes[-40:],
            "eventos_entorno": servicio_sma.monitor.historial_eventos_entorno[-30:],
            "acciones_agentes": servicio_sma.monitor.historial_acciones[-40:],
            "perfiles_clientes": list(
                servicio_sma.entorno.perfilesClientes.perfiles.values()
            ),
            "alertas_humanas": servicio_sma.entorno.alertas_humanas,
            "agenda_virtual": servicio_sma.entorno.agendaVirtual.bloques,
            "bandeja_salida": servicio_sma.entorno.plataformas.bandeja_salida[-25:],
        }

    @app.post("/api/humano/responder")
    async def humano_responder_cliente(req: PeticionRespuestaHumana) -> Dict[str, Any]:
        """Registra y emite una respuesta manual del asesor humano hacia el cliente."""
        return await servicio_sma.responder_como_humano(
            id_cliente=req.id_cliente, texto=req.texto, canal=req.canal
        )

    @app.post("/api/humano/reactivar")
    async def humano_reactivar_cliente(req: PeticionReactivarCliente) -> Dict[str, Any]:
        """Reactiva la atencion automatizada de AgenteVentas para un cliente."""
        perfil = servicio_sma.reactivar_automatizacion(req.id_cliente)
        return {"status": "ok", "perfil": perfil}

    @app.post("/api/calendario/disponibilidad")
    async def cambiar_disponibilidad_calendario(
        req: PeticionCambioDisponibilidad,
    ) -> Dict[str, Any]:
        """Dispara eventoCambioDisponibilidad() desde el panel del profesional humano."""
        return servicio_sma.entorno.eventoCambioDisponibilidad(
            fecha_hora=req.fecha_hora,
            estado=req.estado,
            origen="profesional_humano",
            cliente=req.cliente,
        )

    return app


_HTML_PANEL_CONTROL = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SistemaMultiagenteComercial - Botpress y Monitor FIPA-ACL</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 20px;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 20px;
    }
    h1 { font-size: clamp(1.2rem, 2vw, 1.7rem); color: var(--accent); }
    .subtitulo { color: var(--muted); font-size: 0.9rem; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .card h2 {
      font-size: 1rem;
      color: var(--accent);
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      margin-right: 6px;
      margin-bottom: 4px;
    }
    .ok { background: rgba(34, 197, 94, 0.2); color: var(--green); }
    .warn { background: rgba(245, 158, 11, 0.2); color: var(--amber); }
    .err { background: rgba(239, 68, 68, 0.2); color: var(--red); }
    pre {
      background: #0b1120;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      font-size: 0.8rem;
      overflow-x: auto;
      max-height: 360px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .item {
      border-bottom: 1px solid var(--border);
      padding: 8px 0;
      font-size: 0.85rem;
    }
    .item:last-child { border-bottom: none; }
    input, textarea, select, button {
      width: 100%;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #0b1120;
      color: var(--text);
      margin-bottom: 8px;
      font-size: 0.9rem;
    }
    button {
      background: var(--accent);
      color: #0f172a;
      font-weight: 700;
      cursor: pointer;
      border: none;
    }
    button.sec { background: var(--green); }
    .fila { display: flex; gap: 8px; flex-wrap: wrap; }
    .fila > * { flex: 1; min-width: 140px; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SistemaMultiagenteComercial - Botpress y Monitor FIPA-ACL</h1>
      <div class="subtitulo">Autores: Owen Badel, Maverick Diaz, Jesus Viloria | SPADE + Scikit-Learn + Botpress</div>
    </div>
    <div id="estado-endpoint"></div>
  </header>

  <div class="grid">
    <section class="card">
      <h2>Agentes SPADE y Modelos Scikit-Learn</h2>
      <div id="agentes-box">Cargando...</div>
      <div id="ml-box" style="margin-top: 10px; color: var(--muted); font-size: 0.85rem;"></div>
    </section>

    <section class="card">
      <h2>Consola Human-in-the-Loop</h2>
      <div id="alertas-box" style="margin-bottom: 10px;">Sin alertas activas.</div>
      <input id="hitl-cliente" placeholder="ID cliente (ej. user_01 o 573001234567)" />
      <textarea id="hitl-texto" rows="2" placeholder="Escribe respuesta manual para el cliente..."></textarea>
      <div class="fila">
        <button onclick="enviarRespuestaHumana()">Registrar respuesta humana</button>
        <button class="sec" onclick="reactivarBot()">Reactivar bot automatico</button>
      </div>
      <div id="hitl-msg" style="font-size: 0.8rem; color: var(--muted);"></div>
    </section>
  </div>

  <div class="grid">
    <section class="card">
      <h2>Monitor FIPA-ACL en Tiempo Real</h2>
      <pre id="fipa-box">Esperando mensajes entre agentes...</pre>
    </section>

    <section class="card">
      <h2>EntornoVirtualAtencion y Agenda Virtual</h2>
      <pre id="entorno-box">Esperando eventos...</pre>
    </section>
  </div>

  <script>
    async function cargarEstado() {
      try {
        const res = await fetch('/api/estado');
        const data = await res.json();

        document.getElementById('estado-endpoint').innerHTML =
          '<span class="badge ok">POST /api/botpress/mensaje: ACTIVO</span>' +
          `<span class="badge ${data.api_key_activa ? 'ok' : 'warn'}">${data.api_key_activa ? 'API KEY: ACTIVA' : 'API KEY: ABIERTA (PRUEBAS)'}</span>`;

        document.getElementById('agentes-box').innerHTML = data.agentes
          .map(a => `<div class="item"><strong>${a.nombre}</strong> (${a.rol}) <span class="badge ${a.vivo ? 'ok' : 'err'}">${a.vivo ? 'VIVO' : 'DETENIDO'}</span><br><span style="color:var(--muted)">${a.arquitectura}</span></div>`)
          .join('');

        document.getElementById('ml-box').textContent =
          `RandomForest R2=${data.modelos_ml.random_forest_r2} (${data.modelos_ml.registros_tarifas} tarifas) | KNN R2=${data.modelos_ml.knn_r2_cv} (${data.modelos_ml.registros_ofertas} ofertas)`;

        if (data.alertas_humanas.length === 0) {
          document.getElementById('alertas-box').innerHTML = '<span class="badge ok">Sin escalamientos criticos</span>';
        } else {
          document.getElementById('alertas-box').innerHTML = data.alertas_humanas
            .slice(-5)
            .reverse()
            .map(al => `<div class="item"><span class="badge ${al.resuelta ? 'ok' : 'err'}">${al.resuelta ? 'RESUELTA' : 'PAUSADO'}</span> <strong>${al.id_cliente}</strong> (${al.canal}) - ${al.motivo_critico}</div>`)
            .join('');
        }

        const fipaTexto = data.mensajes_fipa
          .slice()
          .reverse()
          .map(m => `${m.etiqueta_diapositiva}\n${JSON.stringify(m.fipa_json, null, 2)}`)
          .join('\n\n----------------------------------------\n\n');
        document.getElementById('fipa-box').textContent = fipaTexto || 'Sin mensajes FIPA-ACL aun.';

        const agendaStr = Object.entries(data.agenda_virtual)
          .map(([k, v]) => `${k} -> ${v.estado}${v.cliente ? ' (' + v.cliente + ')' : ''}`)
          .join('\n');
        const eventosStr = data.eventos_entorno
          .slice(-10)
          .reverse()
          .map(e => `[${e.timestamp || ''}] ${e.origen}: ${e.detalle}`)
          .join('\n');
        document.getElementById('entorno-box').textContent =
          `AGENDA VIRTUAL:\n${agendaStr}\n\nEVENTOS RECIENTES:\n${eventosStr || 'Ninguno'}`;
      } catch (e) {
        console.error(e);
      }
    }

    async function enviarRespuestaHumana() {
      const id_cliente = document.getElementById('hitl-cliente').value.trim();
      const texto = document.getElementById('hitl-texto').value.trim();
      if (!id_cliente || !texto) return;
      const r = await fetch('/api/humano/responder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id_cliente, texto})
      });
      const j = await r.json();
      document.getElementById('hitl-msg').textContent = 'Respuesta registrada para ' + j.id_cliente + ' (' + j.canal + ')';
      document.getElementById('hitl-texto').value = '';
      cargarEstado();
    }

    async function reactivarBot() {
      const id_cliente = document.getElementById('hitl-cliente').value.trim();
      if (!id_cliente) return;
      await fetch('/api/humano/reactivar', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id_cliente})
      });
      document.getElementById('hitl-msg').textContent = 'Automatizacion reactivada para ' + id_cliente;
      cargarEstado();
    }

    cargarEstado();
    setInterval(cargarEstado, 2500);
  </script>
</body>
</html>
"""
