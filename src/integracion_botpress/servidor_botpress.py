"""
Servidor FastAPI para integracion directa con Botpress (puente hacia WhatsApp,
Webchat, Messenger e Instagram), Monitor FIPA-ACL en vivo y control Human-in-the-Loop.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

import hmac
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.escenarios.orquestador_escenarios import ServicioMultiagenteComercial

_RUTA_MONITOR_HTML = Path(__file__).resolve().parent / "static" / "monitor.html"


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
        return _RUTA_MONITOR_HTML.read_text(encoding="utf-8")

    @app.get("/health")
    async def healthcheck() -> Dict[str, Any]:
        return {
            "status": "ok",
            "servicio": "SistemaMultiagenteComercial-Botpress",
            "modo_bd": servicio_sma.repositorio.modo_bd,
            "negocio_id": servicio_sma.repositorio.negocio_id,
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
        cfg_neg = servicio_sma.entorno.config_negocio
        return {
            "canal_integracion": "Botpress API Bridge",
            "api_key_activa": bool(servicio_sma.config_botpress.botpress_api_key),
            "modo_bd": servicio_sma.repositorio.modo_bd,
            "negocio": {
                "id": servicio_sma.repositorio.negocio_id,
                "nombre": cfg_neg.get("nombre_negocio", "Independiente"),
                "rubro": cfg_neg.get("rubro", "Servicios"),
                "unidad_cantidad": cfg_neg.get("unidad_cantidad", "horas"),
                "moneda": cfg_neg.get("moneda", "COP"),
            },
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

    @app.post("/api/humano/reactivar")
    async def humano_reactivar_cliente(req: PeticionReactivarCliente) -> Dict[str, Any]:
        """Reactiva la atencion automatizada de AgenteVentas tras atencion en WhatsApp."""
        perfil = servicio_sma.reactivar_automatizacion(req.id_cliente)
        return {"status": "ok", "perfil": perfil}

    @app.post("/api/calendario/disponibilidad")
    async def cambiar_disponibilidad_calendario(
        req: PeticionCambioDisponibilidad,
    ) -> Dict[str, Any]:
        """Dispara eventoCambioDisponibilidad() desde el panel de monitoreo."""
        return servicio_sma.entorno.eventoCambioDisponibilidad(
            fecha_hora=req.fecha_hora,
            estado=req.estado,
            origen="profesional_humano",
            cliente=req.cliente,
        )

    return app
