"""
Servidor FastAPI para integracion real con Meta Platforms (WhatsApp Cloud API,
Facebook Messenger e Instagram Direct), verificacion Webhook, Monitor FIPA-ACL
en vivo y consola Human-in-the-Loop.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial


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


def _verificar_firma_meta(body: bytes, signature_header: str, app_secret: str) -> bool:
    """Valida la cabecera X-Hub-Signature-256 firmada por Meta Platforms."""
    if not app_secret:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    esperada = "sha256=" + hmac.new(
        app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(esperada, signature_header)


def extraer_mensajes_payload_meta(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extrae mensajes normalizados desde webhooks oficiales de Meta:
    - whatsapp_business_account (WhatsApp Cloud API)
    - page (Facebook Messenger)
    - instagram (Instagram Direct Messaging)
    """
    mensajes: List[Dict[str, str]] = []
    obj = str(payload.get("object", "")).lower()

    if obj == "whatsapp_business_account":
        for entrada in payload.get("entry", []):
            for cambio in entrada.get("changes", []):
                valor = cambio.get("value", {})
                contactos = {
                    str(c.get("wa_id")): str(c.get("profile", {}).get("name", "Usuario WhatsApp"))
                    for c in valor.get("contacts", [])
                }
                for msg in valor.get("messages", []):
                    wa_id = str(msg.get("from", ""))
                    tipo = str(msg.get("type", ""))
                    texto = ""
                    if tipo == "text":
                        texto = str(msg.get("text", {}).get("body", ""))
                    elif tipo == "interactive":
                        inter = msg.get("interactive", {})
                        if "button_reply" in inter:
                            texto = str(inter["button_reply"].get("title", ""))
                        elif "list_reply" in inter:
                            texto = str(inter["list_reply"].get("title", ""))
                    if wa_id and texto:
                        mensajes.append(
                            {
                                "canal": "WhatsApp",
                                "id_cliente": wa_id,
                                "nombre": contactos.get(wa_id, f"WhatsApp {wa_id}"),
                                "texto": texto,
                            }
                        )

    elif obj in {"page", "instagram"}:
        canal = "Instagram" if obj == "instagram" else "Facebook"
        for entrada in payload.get("entry", []):
            for evento in entrada.get("messaging", []):
                msg = evento.get("message", {})
                if not msg or msg.get("is_echo"):
                    continue
                texto = str(msg.get("text", "")).strip()
                remitente = str(evento.get("sender", {}).get("id", ""))
                if remitente and texto:
                    mensajes.append(
                        {
                            "canal": canal,
                            "id_cliente": remitente,
                            "nombre": f"{canal} {remitente}",
                            "texto": texto,
                        }
                    )

            for cambio in entrada.get("changes", []):
                valor = cambio.get("value", {})
                if cambio.get("field") == "messages":
                    remitente = str(valor.get("sender", {}).get("id", ""))
                    texto = str(valor.get("message", {}).get("text", "")).strip()
                    if remitente and texto:
                        mensajes.append(
                            {
                                "canal": canal,
                                "id_cliente": remitente,
                                "nombre": f"{canal} {remitente}",
                                "texto": texto,
                            }
                        )

    return mensajes


def crear_app_meta(servicio_sma: ServicioMultiagenteComercial) -> FastAPI:
    """Construye la aplicacion FastAPI conectada al SistemaMultiagenteComercial."""
    app = FastAPI(
        title="SistemaMultiagenteComercial - Webhook Meta y Panel FIPA-ACL",
        version="1.0.0",
    )

    @app.get("/", response_class=HTMLResponse)
    async def panel_principal() -> str:
        return _HTML_PANEL_CONTROL

    @app.get("/webhook")
    async def verificar_webhook_meta(
        hub_mode: str = Query(default="", alias="hub.mode"),
        hub_verify_token: str = Query(default="", alias="hub.verify_token"),
        hub_challenge: str = Query(default="", alias="hub.challenge"),
    ) -> PlainTextResponse:
        """Endpoint GET requerido por Meta for Developers para verificar el Webhook."""
        if (
            hub_mode == "subscribe"
            and hub_verify_token == servicio_sma.config_meta.meta_verify_token
        ):
            return PlainTextResponse(content=hub_challenge, status_code=200)
        raise HTTPException(status_code=403, detail="Token de verificacion Meta invalido")

    @app.post("/webhook")
    async def recibir_webhook_meta(request: Request) -> Dict[str, Any]:
        """
        Endpoint POST oficial que recibe mensajes reales en tiempo real desde
        WhatsApp Cloud API, Facebook Messenger e Instagram Direct.
        """
        raw_body = await request.body()
        firma = request.headers.get("X-Hub-Signature-256", "")
        if not _verificar_firma_meta(
            raw_body, firma, servicio_sma.config_meta.meta_app_secret
        ):
            raise HTTPException(status_code=403, detail="Firma X-Hub-Signature-256 invalida")

        payload = await request.json()
        entrantes = extraer_mensajes_payload_meta(payload)
        resultados: List[Dict[str, Any]] = []

        for item in entrantes:
            res = await servicio_sma.procesar_mensaje_entrante_meta(
                canal=item["canal"],
                id_cliente=item["id_cliente"],
                nombre_cliente=item["nombre"],
                texto=item["texto"],
            )
            resultados.append(res)

        return {
            "status": "ok",
            "mensajes_procesados": len(resultados),
            "resultados": resultados,
        }

    @app.get("/api/estado")
    async def obtener_estado_sistema() -> Dict[str, Any]:
        """Retorna el estado completo en vivo del SMA, FIPA-ACL y Human-in-the-Loop."""
        return {
            "canales_meta": servicio_sma.config_meta.estado_canales(),
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
        """Envia una respuesta manual del profesional humano hacia Meta."""
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
  <title>SistemaMultiagenteComercial - Monitor FIPA-ACL y Meta</title>
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
    }
    .ok { background: rgba(34,197,94,0.18); color: var(--green); }
    .warn { background: rgba(245,158,11,0.18); color: var(--amber); }
    .err { background: rgba(239,68,68,0.18); color: var(--red); }
    .fila {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid rgba(51,65,85,0.5);
      font-size: 0.88rem;
    }
    .log-box {
      max-height: 440px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .fipa-item {
      background: #0b1120;
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.8rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    label { display: block; font-size: 0.82rem; color: var(--muted); margin: 8px 0 4px; }
    input, select, textarea {
      width: 100%;
      background: #0b1120;
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 0.88rem;
    }
    button {
      margin-top: 10px;
      background: var(--accent);
      color: #0f172a;
      border: none;
      border-radius: 6px;
      padding: 9px 14px;
      font-weight: 600;
      cursor: pointer;
      min-height: 40px;
    }
    button.sec { background: #334155; color: var(--text); }
    button.btn-sm { padding: 5px 10px; min-height: 32px; font-size: 0.78rem; margin-top: 4px; }
    .acciones-rapidas { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SistemaMultiagenteComercial</h1>
      <div class="subtitulo">MAS-ML | SPADE XMPP | FIPA-ACL | Scikit-Learn | Meta Platforms (WhatsApp, Instagram, Messenger)</div>
      <div class="subtitulo">Autores: Owen Badel, Maverick Diaz, Jesus Viloria | Inteligencia Artificial</div>
    </div>
    <div>
      <span class="badge ok" id="badge-servidor">Webhook activo en /webhook</span>
    </div>
  </header>

  <section class="grid">
    <div class="card">
      <h2>Agentes SPADE y Modelos ML</h2>
      <div id="lista-agentes"></div>
      <div id="resumen-ml" style="margin-top: 10px; font-size: 0.84rem; color: var(--muted);"></div>
    </div>

    <div class="card">
      <h2>Conectores Meta Platforms (.env)</h2>
      <div id="lista-canales"></div>
      <p style="margin-top: 10px; font-size: 0.82rem; color: var(--muted);">
        URL de Webhook para Meta for Developers: <code>/webhook</code>
      </p>
    </div>

    <div class="card">
      <h2>Consola Webhook Meta (Pruebas en Tiempo Real)</h2>
      <label for="canal-sim">Canal de origen (formato payload oficial Meta)</label>
      <select id="canal-sim">
        <option value="WhatsApp">WhatsApp Cloud API</option>
        <option value="Instagram">Instagram Direct</option>
        <option value="Facebook">Facebook Messenger</option>
      </select>
      <label for="cliente-sim">ID / Telefono del cliente</label>
      <input id="cliente-sim" value="573001112233" />
      <label for="texto-sim">Mensaje entrante del cliente</label>
      <textarea id="texto-sim" rows="2">Necesito cotizar un ecommerce complejidad 3 con 45 horas y urgencia 12 dias</textarea>
      <button onclick="enviarWebhookPrueba()">Enviar al Webhook /webhook</button>
      <div class="acciones-rapidas">
        <button class="sec btn-sm" onclick="cargarEjemplo('Hola, que servicios ofrecen?')">1. Consulta Inicial</button>
        <button class="sec btn-sm" onclick="cargarEjemplo('Necesito cotizar un ecommerce complejidad 3 con 45 horas y urgencia 12 dias')">1A. Cotizacion OK</button>
        <button class="sec btn-sm" onclick="cargarEjemplo('Quiero cotizar un desarrollo web')">1B. Faltan Datos</button>
        <button class="sec btn-sm" onclick="cargarEjemplo('El precio me parece un poco costoso, tienen descuento?')">2A. Promo KNN OK</button>
        <button class="sec btn-sm" onclick="cargarEjemplo('Demasiado caro, necesito un 22% de descuento o no cierro')">2B. Promo Rechazada</button>
        <button class="sec btn-sm" onclick="cargarEjemplo('Acepto la cotizacion, quiero agendar el 2026-11-18 10:00')">3. Agendar Cita</button>
        <button class="sec btn-sm" onclick="cargarEjemplo('Esto es un robo, pesimo servicio, pasenme con el dueno ya')">4. Escalar Humano</button>
      </div>
    </div>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Monitor FIPA-ACL en Tiempo Real (Etiqueta + JSON)</h2>
      <div class="log-box" id="monitor-fipa"></div>
    </div>

    <div class="card">
      <h2>Human-in-the-Loop y Respuestas al Cliente</h2>
      <div id="alertas-humanas" style="margin-bottom: 12px;"></div>
      <label for="hitl-cliente">ID Cliente para responder manualmente</label>
      <input id="hitl-cliente" value="573001112233" />
      <label for="hitl-texto">Mensaje del profesional humano hacia Meta</label>
      <textarea id="hitl-texto" rows="2" placeholder="Hola, te atiende directamente el titular del servicio..."></textarea>
      <div style="display:flex; gap:8px;">
        <button onclick="responderHumano()">Enviar Respuesta Humana</button>
        <button class="sec" onclick="reactivarBot()">Reactivar AgenteVentas</button>
      </div>
      <h2 style="margin-top:16px;">Bandeja de Salida hacia Meta</h2>
      <div class="log-box" style="max-height:200px;" id="bandeja-salida"></div>
    </div>
  </section>

  <script>
    function cargarEjemplo(txt) {
      document.getElementById("texto-sim").value = txt;
    }

    function construirPayloadMeta(canal, idCliente, texto) {
      if (canal === "WhatsApp") {
        return {
          object: "whatsapp_business_account",
          entry: [{
            id: "waba_demo",
            changes: [{
              field: "messages",
              value: {
                messaging_product: "whatsapp",
                contacts: [{ profile: { name: "Cliente Prueba" }, wa_id: idCliente }],
                messages: [{ from: idCliente, id: "wamid." + Date.now(), type: "text", text: { body: texto } }]
              }
            }]
          }]
        };
      }
      return {
        object: canal === "Instagram" ? "instagram" : "page",
        entry: [{
          id: "page_demo",
          messaging: [{
            sender: { id: idCliente },
            recipient: { id: "page_001" },
            message: { mid: "mid." + Date.now(), text: texto }
          }]
        }]
      };
    }

    async function enviarWebhookPrueba() {
      const canal = document.getElementById("canal-sim").value;
      const idCliente = document.getElementById("cliente-sim").value.trim();
      const texto = document.getElementById("texto-sim").value.trim();
      const payload = construirPayloadMeta(canal, idCliente, texto);
      await fetch("/webhook", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      await actualizarEstado();
    }

    async function responderHumano() {
      const id_cliente = document.getElementById("hitl-cliente").value.trim();
      const texto = document.getElementById("hitl-texto").value.trim();
      if (!id_cliente || !texto) return;
      await fetch("/api/humano/responder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_cliente, texto })
      });
      document.getElementById("hitl-texto").value = "";
      await actualizarEstado();
    }

    async function reactivarBot() {
      const id_cliente = document.getElementById("hitl-cliente").value.trim();
      if (!id_cliente) return;
      await fetch("/api/humano/reactivar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_cliente })
      });
      await actualizarEstado();
    }

    async function actualizarEstado() {
      const res = await fetch("/api/estado");
      const data = await res.json();

      document.getElementById("lista-agentes").innerHTML = data.agentes.map(a => `
        <div class="fila">
          <span><strong>${a.nombre}</strong> (${a.rol})</span>
          <span class="badge ok">ACTIVO</span>
        </div>
      `).join("");

      document.getElementById("resumen-ml").textContent =
        `RandomForest R2=${data.modelos_ml.random_forest_r2} (${data.modelos_ml.registros_tarifas} reg.) | ` +
        `KNN R2 CV=${data.modelos_ml.knn_r2_cv} (${data.modelos_ml.registros_ofertas} reg.)`;

      document.getElementById("lista-canales").innerHTML = Object.entries(data.canales_meta).map(([k, v]) => `
        <div class="fila">
          <span>${k}</span>
          <span class="badge ${v ? "ok" : "warn"}">${v ? "CONECTADO" : "FALTA TOKEN (.env)"}</span>
        </div>
      `).join("");

      const fipaHtml = [...data.mensajes_fipa].reverse().map(m =>
        `<div class="fipa-item">${m.etiqueta_diapositiva}\n----------------------------------------\n${JSON.stringify(m.fipa_json, null, 2)}</div>`
      ).join("");
      document.getElementById("monitor-fipa").innerHTML = fipaHtml || "<div style='color:var(--muted)'>Sin mensajes FIPA-ACL aun.</div>";

      const alertasHtml = data.alertas_humanas.map(al => `
        <div class="fila">
          <span><strong>${al.id_cliente}</strong> (${al.canal}): ${al.motivo_critico}</span>
          <span class="badge ${al.resuelta ? "ok" : "err"}">${al.resuelta ? "RESUELTA" : "REQUIERE HUMANO"}</span>
        </div>
      `).join("");
      document.getElementById("alertas-humanas").innerHTML = alertasHtml || "<div style='color:var(--muted);font-size:0.84rem;'>Sin alertas criticas pendientes.</div>";

      const salidaHtml = [...data.bandeja_salida].reverse().map(s => `
        <div class="fipa-item">[${s.canal} -> ${s.id_cliente}] (${s.accion})\n${s.contenido}</div>
      `).join("");
      document.getElementById("bandeja-salida").innerHTML = salidaHtml || "<div style='color:var(--muted)'>Sin respuestas enviadas aun.</div>";
    }

    actualizarEstado();
    setInterval(actualizarEstado, 2500);
  </script>
</body>
</html>
"""

