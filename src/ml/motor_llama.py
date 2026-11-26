"""
Motor generativo local basado en Meta Llama ejecutado sobre Ollama para AgenteVentas.

Responsabilidades:
1. Aprovisionamiento automatico en equipos nuevos (asegurar_entorno_llama):
   detecta o instala Ollama, inicia el demonio local (127.0.0.1:11434), selecciona
   el modelo Llama adecuado al hardware (llama3.1:8b en GPU >= 6 GB VRAM o
   llama3.2:3b en CPU/equipos ligeros) y descarga el modelo si aun no existe.
2. Comprension de Lenguaje Natural (NLU) con Salidas Estructuradas (JSON Schema):
   interpreta intencion comercial, tono del cliente, nivel de indecision, alertas
   criticas y parametros del catalogo real de BD usando contexto conversacional
   directamente con el modelo generativo local Meta Llama.
3. Generacion de Lenguaje Natural (NLG) local:
   redacta respuestas comerciales claras y naturales validando estrictamente
   que no se alteren montos, porcentajes ni fechas verificadas.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover - respaldo defensivo si falta el paquete
    OllamaClient = None  # type: ignore[assignment]

from src.ml.datasets import CATALOGO_SERVICIOS
from src.ml.modelos_ml import (
    _NUMEROS_ES,
    _contiene_termino,
    _es_objecion_precio_explicita,
    _extraer_fecha_hora,
    _extraer_hora,
    _normalizar_texto_espanol,
    _numero_espanol_a_entero,
    _variantes_unidad_comercial,
    campo_cantidad_para_unidad,
    normalizar_unidad_comercial,
    rango_complejidad_servicio,
)

_PATRON_MONTO = re.compile(r"\$\s*[\d.,]+")
_NUM_CTX_LLAMA = 4096

_PALABRAS_HOSTILES_SEGURIDAD = (
    "robo",
    "estafa",
    "estafadores",
    "estafador",
    "ladrones",
    "ladron",
    "denunciar",
    "denuncia",
    "demanda",
    "demandar",
    "superintendencia",
    "pesimo",
    "horrible",
    "porqueria",
    "inaceptable",
    "basura",
    "mierda",
    "asco",
    "idiota",
    "idiotas",
    "imbecil",
    "estupido",
    "estupida",
    "maldito",
    "maldita",
    "te odio",
    "odio",
    "verguenza",
    "harto",
    "furioso",
    "furiosa",
    "engano",
    "enganado",
    "enganada",
    "devolucion",
    "devuelvan mi dinero",
    "devuelvan mi plata",
    "hablar con un humano",
    "hablar con una persona",
    "persona real",
    "asesor humano",
    "atencion humana",
    "pasame con un humano",
    "pasenme con un humano",
    "pasame con el dueno",
    "pasenme con el dueno",
    "hablar con el dueno",
    "comunicame con el dueno",
    "con el encargado",
    "con el gerente",
    "bot inutil",
    "no sirve para nada",
)

_PALABRAS_RECHAZO = (
    "no voy a comprar",
    "no me interesa",
    "rechazo la oferta",
    "declino",
    "no deseo seguir",
    "no deseo continuar",
    "no quiero continuar",
    "desisto",
    "cancelo la solicitud",
    "cancelo la cotizacion",
    "cancelo todo",
    "mejor lo dejo asi",
    "por ahora no gracias",
    "deja asi",
    "dejalo asi",
    "ya no lo necesito",
    "no voy a agendar",
    "no voy a contratar",
    "no lo quiero",
    "paso por ahora",
)

_PALABRAS_AGENDA = (
    "agendar",
    "agenda",
    "agendemos",
    "agendame",
    "reservar",
    "reservemos",
    "reserva",
    "cita",
    "horario",
    "calendario",
    "apartar",
    "aparta",
    "cupo",
    "turno",
    "disponibilidad",
    "acepto",
    "apruebo",
    "aprobado",
    "cerramos trato",
    "separemos",
    "separa el cupo",
    "programar",
)

_ESQUEMA_NLU_LLAMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intencion": {
            "type": "string",
            "enum": [
                "consulta_servicio",
                "consulta_sin_informacion",
                "solicitud_cotizacion",
                "parametros_servicio",
                "objecion_precio",
                "solicitud_agendamiento",
                "aceptar_cotizacion",
                "rechazo_compra",
                "reclamo_critico",
            ],
        },
        "es_saludo_simple": {"type": "boolean"},
        "es_pregunta_catalogo": {"type": "boolean"},
        "cotizacion_explicita": {"type": "boolean"},
        "informacion_no_disponible": {"type": "boolean"},
        "tono_cliente": {
            "type": "string",
            "enum": ["interesado", "indeciso", "hostil"],
        },
        "nivel_indecision": {"type": "number"},
        "escalamiento_humano": {"type": "boolean"},
        "tipo_servicio": {"type": ["string", "null"]},
        "cantidad_unidades": {"type": ["integer", "null"]},
        "complejidad": {"type": ["integer", "null"]},
        "urgencia_dias": {"type": ["integer", "null"]},
        "incluye_adicional": {"type": "boolean"},
        "descuento_solicitado_cliente": {"type": ["number", "null"]},
        "es_personalizado": {"type": "boolean"},
        "descripcion_personalizada": {"type": ["string", "null"]},
        "cantidad_fotos": {"type": ["integer", "null"]},
        "cantidad_personas": {"type": ["integer", "null"]},
        "en_exterior": {"type": ["boolean", "null"]},
        "fuera_de_alcance": {"type": "boolean"},
        "fecha_hora_texto": {"type": ["string", "null"]},
    },
    "required": [
        "intencion",
        "es_saludo_simple",
        "es_pregunta_catalogo",
        "cotizacion_explicita",
        "informacion_no_disponible",
        "tono_cliente",
        "nivel_indecision",
        "escalamiento_humano",
        "tipo_servicio",
        "cantidad_unidades",
        "complejidad",
        "urgencia_dias",
        "incluye_adicional",
        "descuento_solicitado_cliente",
        "es_personalizado",
        "descripcion_personalizada",
        "cantidad_fotos",
        "cantidad_personas",
        "en_exterior",
        "fuera_de_alcance",
        "fecha_hora_texto",
    ],
}

_INSTRUCCION_SISTEMA_NLG = (
    "Eres el asesor comercial de un negocio profesional atendiendo clientes por chat. "
    "Redacta respuestas claras, cercanas, naturales y concisas en espanol "
    "(maximo 4 oraciones o lista corta si muestras un catalogo). "
    "Reglas estrictas e inviolables:\n"
    "1. Nunca inventes precios, descuentos, porcentajes, servicios, direcciones, ubicaciones, "
    "metodos de pago, tiempos de entrega, politicas, equipos, garantias ni fechas/horarios "
    "que no figuren literalmente en los datos verificados.\n"
    "2. Conserva de forma exacta todos los montos en dinero ($...), porcentajes (%) "
    "y fechas/horas suministrados.\n"
    "3. Nunca menciones tecnicismos internos como 'parametros obligatorios', 'borrador', 'segun el borrador', "
    "'FIPA', 'SPADE', 'Random Forest', 'KNN', 'Ollama', 'Llama', 'base de datos' ni nombres de variables.\n"
    "4. Nunca digas 'no puedo proporcionar la respuesta final' ni 'no tengo la informacion' cuando el texto base si te da una pregunta o respuesta; reformula el texto base de forma amable.\n"
    "5. Nunca prometas 'fotos ilimitadas' ni 'sin limite fijo de fotos' si el texto base no lo afirma; "
    "si el cliente pregunta cuantas fotos incluye una cobertura estandar, aclara que la entrega depende "
    "del alcance acordado de la sesion y que tambien se puede cotizar un paquete a medida por cantidad exacta de fotos.\n"
    "6. No uses emojis.\n"
    "7. Responde directamente al cliente sin preambulos ni comillas envolventes.\n"
    "8. No inventes preguntas sobre requisitos que no figuren en el texto base.\n"
    "9. Nunca contradigas el texto base ni digas 'no tenemos' o 'no ofrecemos' un servicio cuando el texto base esta ofreciendo ayuda o pidiendo datos."
)


def _normalizar_comparacion(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.casefold())
    sin_tildes = "".join(
        c for c in descompuesto if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_tildes).strip()


def _extraer_montos_normalizados(texto: str) -> set[str]:
    return {
        re.sub(r"\s+", "", m.group(0)).rstrip(".,")
        for m in _PATRON_MONTO.finditer(texto or "")
    }


def _detectar_vram_gpu_mib() -> int:
    """Detecta la VRAM disponible en GPU NVIDIA (en MiB) o 0 si solo hay CPU."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        salida = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        valores = [
            int(linea.strip())
            for linea in salida.splitlines()
            if linea.strip().isdigit()
        ]
        return max(valores) if valores else 0
    except Exception:
        return 0


def resolver_modelo_llama(modelo_configurado: str = "auto") -> str:
    """
    Resuelve el modelo Meta Llama optimo para el equipo actual.
    - 'auto': usa llama3.1:8b si hay GPU >= 6 GB VRAM; de lo contrario llama3.2:3b.
    - Cualquier otro nombre explicito (ej. 'llama3.1:8b' o 'llama3.2:3b') se respeta.
    """
    valor = (modelo_configurado or "auto").strip()
    if valor.lower() != "auto":
        return valor
    vram_mib = _detectar_vram_gpu_mib()
    return "llama3.1:8b" if vram_mib >= 6000 else "llama3.2:3b"


def _localizar_binario_ollama() -> Optional[str]:
    """Busca el ejecutable de Ollama en PATH y en rutas estandar de Windows/macOS/Linux."""
    en_path = shutil.which("ollama")
    if en_path:
        return en_path
    if platform.system() == "Windows":
        candidatos = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        ]
        for candidato in candidatos:
            if candidato.is_file():
                return str(candidato)
    return None


def _es_host_local(host: str) -> bool:
    """Retorna True si la URL de Ollama apunta a esta misma maquina."""
    h = (host or "").strip().casefold()
    return any(tok in h for tok in ("127.0.0.1", "localhost", "[::1]", "0.0.0.0"))


def _ollama_responde(host: str, timeout: float = 2.5) -> bool:
    """Verifica si el servidor HTTP de Ollama (local o remoto) responde en /api/tags."""
    try:
        resp = httpx.get(
            f"{host.rstrip('/')}/api/tags",
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _instalar_ollama_sistema(consola: Optional[Any] = None) -> Optional[str]:
    """
    Instala el runtime local Ollama automaticamente en un computador nuevo
    (Windows via winget o instalador oficial silencioso; macOS via brew; Linux via script).
    """
    sistema = platform.system()
    if consola is not None:
        consola.print("[cyan]Ollama no detectado. Instalando runtime local Llama automaticamente...[/cyan]")

    if sistema == "Windows":
        if shutil.which("winget"):
            try:
                subprocess.run(
                    [
                        "winget",
                        "install",
                        "-e",
                        "--id",
                        "Ollama.Ollama",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                        "--silent",
                    ],
                    check=False,
                    timeout=300,
                )
            except Exception:
                pass
        binario = _localizar_binario_ollama()
        if binario:
            return binario
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                instalador = Path(tmp_dir) / "OllamaSetup.exe"
                with httpx.stream(
                    "GET",
                    "https://ollama.com/download/OllamaSetup.exe",
                    follow_redirects=True,
                    timeout=180.0,
                ) as resp:
                    resp.raise_for_status()
                    with instalador.open("wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
                subprocess.run(
                    [str(instalador), "/VERYSILENT", "/NORESTART"],
                    check=False,
                    timeout=300,
                )
        except Exception:
            pass
        return _localizar_binario_ollama()

    if sistema == "Darwin" and shutil.which("brew"):
        try:
            subprocess.run(["brew", "install", "ollama"], check=False, timeout=300)
        except Exception:
            pass
        return _localizar_binario_ollama()

    if sistema == "Linux" and shutil.which("curl"):
        try:
            subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=False,
                timeout=300,
            )
        except Exception:
            pass
        return _localizar_binario_ollama()

    return _localizar_binario_ollama()


def _iniciar_servidor_ollama(binario: str, host: str) -> bool:
    """Inicia `ollama serve` en segundo plano si aun no escucha localmente."""
    if _ollama_responde(host):
        return True
    try:
        kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([binario, "serve"], **kwargs)
    except Exception:
        return False

    for _ in range(25):
        if _ollama_responde(host, timeout=1.5):
            return True
        time.sleep(0.5)
    return _ollama_responde(host, timeout=2.0)


def _modelo_instalado_en_ollama(cliente: Any, modelo: str) -> bool:
    """Comprueba si el modelo Llama solicitado ya esta descargado en Ollama."""
    try:
        listado = cliente.list()
    except Exception:
        return False
    modelos_raw = getattr(listado, "models", None)
    if modelos_raw is None and isinstance(listado, dict):
        modelos_raw = listado.get("models", [])
    objetivo = modelo.strip().casefold()
    objetivo_base = objetivo.split(":")[0]
    for item in modelos_raw or []:
        nombre = getattr(item, "model", None) or (
            item.get("model") or item.get("name") if isinstance(item, dict) else ""
        )
        nombre_norm = str(nombre or "").strip().casefold()
        if nombre_norm == objetivo or (
            ":" not in objetivo and nombre_norm.split(":")[0] == objetivo_base
        ):
            return True
    return False


def asegurar_entorno_llama(
    host: str = "http://127.0.0.1:11434",
    modelo: str = "auto",
    auto_instalar: bool = True,
    consola: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Asegura que el runtime local Ollama y el modelo Meta Llama esten listos
    antes de iniciar los agentes SPADE. En un computador nuevo instala Ollama
    y descarga el modelo automaticamente si `auto_instalar=True`.
    """
    es_local = _es_host_local(host)
    modelo_resuelto = resolver_modelo_llama(modelo)
    resultado: Dict[str, Any] = {
        "disponible": False,
        "host": host,
        "modelo": modelo_resuelto,
        "auto_instalado": False,
        "detalle": "",
    }
    if OllamaClient is None:
        resultado["detalle"] = "Paquete Python 'ollama' no instalado."
        return resultado

    servidor_listo = _ollama_responde(host)
    if not servidor_listo and es_local:
        binario = _localizar_binario_ollama()
        if not binario and auto_instalar:
            binario = _instalar_ollama_sistema(consola=consola)
            if binario:
                resultado["auto_instalado"] = True
        if binario:
            servidor_listo = _iniciar_servidor_ollama(binario, host)

    if not servidor_listo:
        resultado["detalle"] = (
            f"No fue posible conectar con Ollama en {host}. "
            + (
                "Verifica la instalacion de Ollama o inicia 'ollama serve'."
                if es_local
                else "Verifica que el equipo con GPU tenga Ollama y el tunel/red activos."
            )
        )
        return resultado

    cliente = OllamaClient(
        host=host,
        timeout=120.0,
        headers={"ngrok-skip-browser-warning": "true"},
    )
    if not es_local and (modelo or "auto").strip().lower() == "auto":
        try:
            listado = cliente.list()
            modelos_raw = getattr(listado, "models", None)
            if modelos_raw is None and isinstance(listado, dict):
                modelos_raw = listado.get("models", [])
            nombres = [
                str(
                    getattr(item, "model", None)
                    or (item.get("model") or item.get("name") if isinstance(item, dict) else "")
                    or ""
                ).strip()
                for item in (modelos_raw or [])
            ]
            nombres = [n for n in nombres if n]
            if any(n.casefold() == "llama3.1:8b" for n in nombres):
                modelo_resuelto = "llama3.1:8b"
            elif nombres:
                modelo_resuelto = nombres[0]
            resultado["modelo"] = modelo_resuelto
        except Exception:
            pass

    if not _modelo_instalado_en_ollama(cliente, modelo_resuelto):
        if not auto_instalar or not es_local:
            resultado["detalle"] = f"Modelo '{modelo_resuelto}' no descargado en Ollama ({host})."
            return resultado
        if consola is not None:
            consola.print(
                f"[cyan]Descargando modelo generativo local '{modelo_resuelto}' en Ollama (solo primera vez)...[/cyan]"
            )
        try:
            for evento in cliente.pull(modelo_resuelto, stream=True):
                estado = getattr(evento, "status", None) or (
                    evento.get("status") if isinstance(evento, dict) else ""
                )
                if estado == "success":
                    break
            resultado["auto_instalado"] = True
        except Exception as exc:
            resultado["detalle"] = f"Fallo al descargar '{modelo_resuelto}': {exc}"
            return resultado

    # Precalentamiento en VRAM/RAM con el mismo num_ctx para evitar recargas
    try:
        cliente.chat(
            model=modelo_resuelto,
            messages=[{"role": "user", "content": "ok"}],
            options={"num_predict": 1, "temperature": 0.0, "num_ctx": _NUM_CTX_LLAMA},
            keep_alive="30m",
        )
    except Exception:
        pass

    resultado["disponible"] = True
    resultado["detalle"] = f"Ollama operativo con modelo {modelo_resuelto}"
    return resultado


def _es_pregunta_catalogo(texto_norm: str) -> bool:
    """Detecta consultas informativas sobre que servicios o paquetes ofrece el negocio."""
    patrones = (
        r"\b(?:que|cuales)\s+(?:servicios|paquetes|planes|productos|opciones|especialidades)\b",
        r"\bq\s+(?:servicios|paquetes|planes|productos)\b",
        r"\b(?:informacion|info)\s+(?:general\s+)?(?:sobre|de)\s+(?:sus\s+|los\s+)?servicios\b",
        r"\b(?:conocer|ver|mostrar|muestrame|enviame|envias|mandame|pasame|comparteme)\s+"
        r"(?:el\s+|la\s+|los\s+|las\s+|un\s+|su\s+|sus\s+|tu\s+|tus\s+)?"
        r"(?:catalogo|portafolio|servicios|paquetes|opciones)\b",
        r"\b(?:te\s+)?(?:pedi|pido|solicite|solicito|necesito|quiero)\s+(?:ver\s+)?"
        r"(?:el\s+|su\s+|tu\s+|un\s+)?(?:catalogo|portafolio)\b",
        r"\b(?:su|tu|el|un)\s+(?:catalogo|portafolio)(?:\s+de\s+servicios?)?\b",
        r"\b(?:que\s+es\s+lo\s+que\s+ofrecen|que\s+ofrecen|que\s+tienen|que\s+manejan|que\s+venden|que\s+vendes|a\s+que\s+se\s+dedican)\b",
        r"\b(?:tienen|manejan|cuentan\s+con)\s+(?:un\s+)?(?:catalogo|portafolio)\b",
    )
    return any(re.search(patron, texto_norm) for patron in patrones)


_PATRONES_INFO_NO_DISPONIBLE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # Ubicacion, direccion, ciudad, cobertura geografica, envios, instalaciones
    (r"\bdonde\s+(?:estan|quedan?|se\s+encuentran?|queda\s+el\s+estudio|queda\s+la\s+oficina)\b", ()),
    (r"\b(?:cual\s+es\s+(?:la|su)\s+)?direccion(?:\s+(?:exacta|del\s+estudio|fisica))?\b", ("direccion",)),
    (r"\b(?:ubicacion|en\s+que\s+ciudad|en\s+que\s+barrio|en\s+que\s+zona)\b", ("ubicacion",)),
    (r"\b(?:trabajan|atienden|viajan|cubren)\s+en\s+(?:otra|otras)\s+ciudades?\b", ()),
    (r"\b(?:hacen\s+envios?|envios?\s+internacionales?|domicilio\s+a\s+otra\s+ciudad)\b", ("envio",)),
    (r"\b(?:tienen|donde\s+esta\s+la)\s+(?:sede|sucursal|tienda\s+fisica|parqueadero)\b", ("parqueadero", "sucursal")),
    # Metodos de pago, anticipos, cuotas, impuestos, facturacion
    (r"\b(?:metodos?|medios?|formas?)\s+de\s+pago\b", ("metodo de pago", "medios de pago")),
    (r"\bcomo\s+(?:se\s+paga|puedo\s+pagar|reciben\s+el\s+pago)\b", ()),
    (r"\b(?:aceptan|reciben|manejan|se\s+puede\s+pagar\s+con)\s+(?:tarjeta|credito|debito|nequi|daviplata|efectivo|transferencia|bancolombia|pse|paypal|cripto|criptomonedas|bitcoin|dolares|usdt)\b", ("nequi", "daviplata", "paypal", "pse", "cripto")),
    (r"\b(?:nequi|daviplata|paypal|bancolombia|criptomonedas?|bitcoin)\b", ("nequi", "daviplata", "paypal")),
    (r"\b(?:pago\s+a|pagar\s+en|a|en)\s+cuotas\b|\bfinanciacion\b", ("cuotas", "financiacion")),
    (r"\b(?:anticipo|abono\s+inicial|hay\s+que\s+abonar|cuanto\s+(?:es\s+el\s+)?(?:anticipo|abono))\b", ("anticipo", "abono")),
    (r"\b(?:emiten|dan|generan|hacen|entregan)\s+factura\b|\bfactura\s+electronica\b", ("factura",)),
    (r"\b(?:cual\s+es\s+(?:el|su)\s+nit|numero\s+de\s+nit|tienen\s+rut|incluye\s+iva|mas\s+iva|con\s+iva|cuenta\s+bancaria)\b", ("nit", "rut", "iva")),
    # Tiempos de entrega, formatos fisicos/RAW, politicas, garantias, contratos
    (r"\b(?:en\s+)?cuant[oa]s?\s+(?:dias?|horas?|semanas?|tiempo)\s+(?:se\s+)?(?:demoran?|tardan?|entregan?)\b", ()),
    (r"\b(?:tiempo|plazo|fecha)\s+de\s+entrega\b|\bcuando\s+entregan\b", ("tiempo de entrega",)),
    (r"\b(?:entregan|dan|envian|incluye)\s+(?:las\s+)?(?:fotos\s+)?(?:en\s+)?(?:raw|sin\s+editar|impresas|fisicas)\b|\barchivos?\s+raw\b", ("raw", "impresas")),
    (r"\b(?:impresion\s+fisica|fotos\s+impresas|album\s+impreso|album\s+fisico|retablo|memoria\s+usb|\busb\b)\b", ("album", "impreso", "usb")),
    (r"\b(?:tienen|cual\s+es\s+la)\s+garantia\b|\bgarantia\s+de\b", ("garantia",)),
    (r"\bpolitica\s+de\s+(?:cancelacion|reembolso|devolucion|cambios?)\b|\bque\s+pasa\s+si\s+(?:llueve|cancelo|reprogramo)\b", ()),
    (r"\b(?:derechos?\s+de\s+autor|(?:firman|hacen|hay)\s+contrato|clausulas?)\b", ("derechos de autor", "contrato")),
    # Equipos, personal, extras fuera de catalogo, horario de apertura
    (r"\bque\s+(?:camara|camaras|lente|lentes|equipo|equipos)\s+(?:usan|utilizan|tienen|manejan)\b", ()),
    (r"\b(?:usan|incluye|tienen|tomas\s+con|con)\s+dron(?:es)?\b|\bdron(?:es)?\b", ("dron", "drone")),
    (r"\b(?:cabina\s+360|photobooth|maquillaje|maquilladora|peinado|vestuario|estilista|modelos?)\b", ("cabina 360", "photobooth", "maquillaje", "vestuario", "peinado", "modelos")),
    (r"\bcuant[oa]s?\s+fotograf[oa]s?\b", ()),
    (r"\bhorarios?\s+de\s+atencion\b|\ba\s+que\s+hora\s+(?:abren|cierran)\b|\b(?:abren|atienden|trabajan)\s+(?:los\s+)?(?:domingos?|festivos?|sabados?|semana\s+santa)\b", ("horario de atencion",)),
    # Redes, portafolio externo, equipo humano, alquiler, convenios
    (r"\b(?:tienen|cual\s+es\s+(?:su|tu|el|la))\s+(?:instagram|tiktok|facebook|portafolio\s+web|pagina\s+web|sitio\s+web)\b", ("instagram", "tiktok", "facebook")),
    (r"\b(?:fotos\s+de\s+muestra|trabajos?\s+anteriores|ejemplos?\s+de\s+trabajo)\b", ()),
    (r"\b(?:cuantos?\s+)?anos\s+de\s+experiencia\b|\bquien\s+es\s+el\s+(?:fotografo|dueno|encargado|gerente)\b", ()),
    (r"\b(?:alquilan|prestan|arriendan)\s+(?:camaras?|equipos?|lentes?|el\s+estudio|luces?)\b|\bconvenios?\b", ("alquiler", "convenio")),
    # Conocimiento general / fuera de dominio
    (r"\b(?:quien\s+es\s+el\s+presidente|cual\s+es\s+la\s+capital|que\s+clima\s+hace|receta\s+de|como\s+se\s+(?:prepara|cocina)|que\s+hora\s+es\s+en|ayudame\s+con\s+mi\s+tarea|cuenta(?:me)?\s+un\s+chiste)\b", ()),
)


def _detectar_consulta_sin_informacion(
    texto_norm: str,
    contexto_negocio: Optional[Dict[str, Any]] = None,
    catalogo_servicios: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Detecta preguntas cuya respuesta no existe en la BD, reglas de negocio ni codigo.
    Respeta dinamicamente lo que si este documentado en el negocio o en servicios.descripcion.
    """
    if not texto_norm:
        return False
    partes_bd: List[str] = []
    if contexto_negocio:
        partes_bd.append(str(contexto_negocio.get("nombre_negocio", "")))
        partes_bd.append(str(contexto_negocio.get("rubro", "")))
        partes_bd.append(str(contexto_negocio.get("mensaje_catalogo_intro", "")))
    if catalogo_servicios:
        for meta in catalogo_servicios.values():
            partes_bd.append(str(meta.get("nombre", "")))
            partes_bd.append(str(meta.get("descripcion", "")))
    texto_bd = _normalizar_texto_espanol(" ".join(partes_bd))

    for patron, palabras_en_bd in _PATRONES_INFO_NO_DISPONIBLE:
        if re.search(patron, texto_norm):
            if palabras_en_bd and any(p in texto_bd for p in palabras_en_bd):
                continue
            return True
    return False


def _es_solicitud_cotizacion_explicita(texto_norm: str) -> bool:
    """Retorna True cuando el cliente solicita explicitamente cotizar o saber precio/valor/costo."""
    if _es_pregunta_catalogo(texto_norm) and "cotiz" not in texto_norm:
        return False
    patrones = (
        r"\bcotiz",
        r"\bpresupuesto\b",
        r"\b(?:en\s+)?cuant[oa]\s+(?:me\s+|nos\s+)?(?:cuesta|costaria|costarian|sale|saldria|salen|vale|valen|valdria|seria|cobran|cobrarian|es|cambiaria|queda)\b",
        r"\ba\s+como\s+(?:es|esta|sale|son|cobran|manejan)\b",
        r"\b(?:que|cual)\s+(?:valor|precio|costo|tarifa)\b",
        r"\b(?:dame|dime|enviame|indica(?:me)?|pasame|cual\s+es)\s+(?:el\s+|un\s+)?(?:valor|precio|costo|presupuesto|tarifa)\b",
        r"\b(?:precio|tarifa|costo|valor)\s+(?:de|del|para|aproximado|estimado|tiene)\b",
        r"\b(?:precio|tarifas)\b",
    )
    return any(re.search(patron, texto_norm) for patron in patrones)


def _es_seleccion_paquete(texto_norm: str) -> bool:
    """Detecta cuando el cliente elige o manifiesta interes por un paquete o servicio."""
    if _es_pregunta_catalogo(texto_norm):
        return False
    return bool(
        re.search(
            r"\b(?:quiero|elijo|escojo|selecciono|prefiero|tomo|dame|me\s+interesa|me\s+gustaria|me\s+quedo\s+con)\s+"
            r"(?:el\s+de\s+|la\s+de\s+|el\s+|la\s+)?(?:ese|esa|eso|este|esta|esto|aquel|aquella|mismo|misma|"
            r"primer[oa]|segund[oa]|tercer[oa]|cuart[oa]|paquete|servicio|opcion)\b"
            r"|\b(?:elijo|escojo|selecciono)\b"
            r"|\b(?:el|la)\s+(?:primer[oa]|segund[oa]|tercer[oa]|cuart[oa])\b"
            r"|\b(?:paquete|opcion)\s*(?:numero\s*|#)?\s*[1-9]\b",
            texto_norm,
        )
    )


class MotorLlamaVentas:
    """
    Motor generativo local (NLU + NLG) de AgenteVentas impulsado por Meta Llama
    sobre Ollama con salidas estructuradas JSON Schema y validador anti-alucinacion.
    """

    def __init__(
        self,
        catalogo_servicios: Optional[Dict[str, Dict[str, Any]]] = None,
        unidad_cantidad: str = "horas",
        etiqueta_cantidad: Optional[str] = None,
        equivalencias_complejidad: Optional[Dict[str, int]] = None,
        rango_complejidad: Tuple[int, int] = (1, 5),
        etiqueta_complejidad: Optional[str] = None,
        contexto_negocio: Optional[Dict[str, Any]] = None,
        modelo: str = "auto",
        host: str = "http://127.0.0.1:11434",
        timeout_segundos: float = 30.0,
        habilitado: bool = True,
        cliente_ollama: Optional[Any] = None,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self.catalogo_servicios: Dict[str, Dict[str, Any]] = (
            catalogo_servicios if catalogo_servicios is not None else CATALOGO_SERVICIOS
        )
        self.unidad_cantidad: str = unidad_cantidad.lower().strip() or "unidades"
        self.etiqueta_cantidad = etiqueta_cantidad
        self.etiqueta_complejidad = etiqueta_complejidad or "complejidad"
        self.equivalencias_complejidad = dict(equivalencias_complejidad or {})
        self.rango_complejidad = rango_complejidad_servicio(
            rango_default=rango_complejidad
        )
        self.contexto_negocio = dict(contexto_negocio or {})
        self.host = (host or "http://127.0.0.1:11434").strip()
        self.modelo = resolver_modelo_llama(modelo)
        self.timeout_segundos = max(1.0, float(timeout_segundos or 30.0))
        self._habilitado = bool(habilitado)
        self._reloj = reloj
        self._bloqueado_hasta: float = 0.0
        self.ultimo_error: str = ""
        self.llamadas_nlu_exitosas: int = 0
        self.llamadas_nlg_exitosas: int = 0
        self.fallbacks_activados: int = 0

        if cliente_ollama is not None:
            self._cliente: Optional[Any] = cliente_ollama
        elif self._habilitado and OllamaClient is not None:
            self._cliente = OllamaClient(
                host=self.host,
                timeout=self.timeout_segundos,
                headers={"ngrok-skip-browser-warning": "true"},
            )
        else:
            self._cliente = None

    @property
    def activo(self) -> bool:
        """Indica si Llama local (Ollama) esta habilitado y con cliente disponible."""
        return self._habilitado and self._cliente is not None

    @property
    def en_enfriamiento(self) -> bool:
        """Indica si el circuit breaker esta abierto tras error de conexion/timeout."""
        return self._reloj() < self._bloqueado_hasta

    def estado_actual(self) -> Dict[str, Any]:
        """Retorna telemetria operativa del motor Llama local para monitoreo y API."""
        if not self.activo:
            modo = "Modo Pruebas Determinista (Ollama desactivado)"
        elif self.en_enfriamiento:
            modo = f"Respaldo Local (Ollama en reconexion: {self.modelo})"
        else:
            modo = f"Llama Local via Ollama ({self.modelo})"
        return {
            "habilitado": self.activo,
            "motor": "Ollama / Meta Llama",
            "modelo": self.modelo,
            "host": self.host,
            "modo": modo,
            "en_enfriamiento": self.en_enfriamiento,
            "llamadas_nlu_exitosas": self.llamadas_nlu_exitosas,
            "llamadas_nlg_exitosas": self.llamadas_nlg_exitosas,
            "fallbacks_activados": self.fallbacks_activados,
            "ultimo_error": self.ultimo_error,
        }

    def _activar_enfriamiento(self, segundos: float, motivo: str) -> None:
        self._bloqueado_hasta = self._reloj() + segundos
        self.ultimo_error = motivo
        self.fallbacks_activados += 1

    # ------------------------------------------------------------------
    # NLU ESTRUCTURADO CON LLAMA LOCAL (OLLAMA STRUCTURED OUTPUTS)
    # ------------------------------------------------------------------
    def _construir_mensajes_nlu(
        self,
        texto: str,
        contexto_tipo: Optional[str],
        campos_pendientes: List[str],
        parametros_acumulados: Dict[str, Any],
        historial_reciente: Optional[List[Dict[str, Any]]],
        estado_conversacion: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        nombre_neg = str(self.contexto_negocio.get("nombre_negocio", "el negocio"))
        rubro = str(self.contexto_negocio.get("rubro", "servicios profesionales"))
        moneda = str(self.contexto_negocio.get("moneda", "COP"))
        lineas_catalogo = [
            f"- clave='{clave}' | nombre='{meta.get('nombre', clave)}' | "
            f"palabras_clave={meta.get('palabras_clave', [])} | descripcion='{meta.get('descripcion', '')}'"
            for clave, meta in self.catalogo_servicios.items()
        ]
        claves_validas = ", ".join(self.catalogo_servicios.keys())

        sistema = (
            f"Analizador NLU de '{nombre_neg}' (rubro: {rubro}, unidad: {self.unidad_cantidad}). "
            "Responde SOLO JSON valido con estas claves:\n"
            "intencion ('consulta_servicio'|'consulta_sin_informacion'|'solicitud_cotizacion'|'parametros_servicio'|'objecion_precio'|"
            "'solicitud_agendamiento'|'aceptar_cotizacion'|'rechazo_compra'|'reclamo_critico'), "
            "es_saludo_simple (bool), es_pregunta_catalogo (bool), cotizacion_explicita (bool), "
            "informacion_no_disponible (bool), "
            "tono_cliente ('interesado'|'indeciso'|'hostil'), nivel_indecision (0..1), escalamiento_humano (bool), "
            f"tipo_servicio (SOLO una de [{claves_validas}] o null), cantidad_unidades (int|null), "
            f"complejidad ({self.rango_complejidad[0]}..{self.rango_complejidad[1]}|null), urgencia_dias (int|null), "
            "incluye_adicional (bool), descuento_solicitado_cliente (number|null), es_personalizado (bool), "
            "descripcion_personalizada (string|null), cantidad_fotos (int|null), cantidad_personas (int|null), "
            "en_exterior (bool|null), fuera_de_alcance (bool), fecha_hora_texto (string|null).\n"
            "Catalogo (UNICA informacion verificada que conoce el sistema sobre servicios):\n" + "\n".join(lineas_catalogo) + "\n"
            "Reglas:\n"
            "- Saludo ('hola','buenas'): intencion='consulta_servicio', es_saludo_simple=true, informacion_no_disponible=false, tipo_servicio=null.\n"
            "- Pedir catalogo/servicios/paquetes ('ver su catalogo','su catalogo de servicio','que ofrecen','que venden','te pedi el catalogo'): "
            "intencion='consulta_servicio', es_pregunta_catalogo=true, informacion_no_disponible=false, tipo_servicio=null, cotizacion_explicita=false.\n"
            "- Pregunta informativa cuya respuesta SI esta en el Catalogo o en la cotizacion ya entregada ('que trae ese paquete','que incluye','cuantas fotos trae','en base a que sacas ese valor'): "
            "intencion='consulta_servicio', es_pregunta_catalogo=false, informacion_no_disponible=false, cotizacion_explicita=false, manteniendo tipo_servicio si menciona un paquete.\n"
            "- CERO INVENCION / ESCALAMIENTO POR DESCONOCIMIENTO: Si el cliente pregunta CUALQUIER dato o politica que NO este literalmente en el Catalogo ni corresponda a cotizar/ajustar un servicio, promocion o agendar cita "
            "(por ejemplo: direccion/ubicacion/ciudad, metodos de pago/Nequi/tarjeta/cuotas/anticipo, factura/NIT/IVA, tiempos de entrega, archivos RAW/impresiones/album, drones, camaras/lentes/equipos, maquillaje/vestuario/modelos, politicas de cancelacion/reembolso/lluvia, horarios de atencion/domingos/festivos, redes/Instagram/fotos de muestra, experiencia/personal, convenios, alquiler de equipos o preguntas generales ajenas al negocio): "
            "DEBES responder intencion='consulta_sin_informacion', informacion_no_disponible=true, escalamiento_humano=true, tono_cliente='interesado', es_saludo_simple=false, es_pregunta_catalogo=false, cotizacion_explicita=false.\n"
            "- Elegir paquete sin pedir precio ('quiero este','sesion para mi boda'): intencion='solicitud_cotizacion', cotizacion_explicita=false, informacion_no_disponible=false. "
            "Mapea bodas/fiestas/grados/bautizos a la clave de eventos del catalogo (ej. 'cobertura_evento') y fotos familiares/pareja/perfil a retrato si existen en el catalogo.\n"
            "- NUNCA asumas cantidad_unidades=1 si el cliente no dijo explicitamente un numero de sesiones/horas/unidades; en ese caso deja cantidad_unidades=null.\n"
            "- Pedir precio/cotizar ('cuanto cuesta','cotizame','que valor tiene'): intencion='solicitud_cotizacion', cotizacion_explicita=true, informacion_no_disponible=false.\n"
            "- Si campos_pendientes_cotizacion tiene datos y responde un dato (ej.'3','2 sesiones'): intencion='parametros_servicio', informacion_no_disponible=false.\n"
            "- Si paso_reserva es 'nombre_cliente','cedula' o 'telefono' y responde con ese dato sin hacer una pregunta externa: informacion_no_disponible=false, escalamiento_humano=false.\n"
            "- Caro/descuento: intencion='objecion_precio', tono_cliente='indeciso', nivel_indecision>=0.72, informacion_no_disponible=false.\n"
            "- Acepta cotizacion o pide cita/fecha: intencion='aceptar_cotizacion' o 'solicitud_agendamiento', informacion_no_disponible=false.\n"
            "- Rechaza compra ('no quiero','cancelo'): intencion='rechazo_compra', informacion_no_disponible=false.\n"
            "- Insulto/estafa/pide humano o dueno: intencion='reclamo_critico', tono_cliente='hostil', escalamiento_humano=true, informacion_no_disponible=false.\n"
            "- Solicitud explicita de cotizar un rubro ajeno (ej. 'cotizar plomeria de 20 horas'): intencion='solicitud_cotizacion', fuera_de_alcance=true, informacion_no_disponible=false, escalamiento_humano=false, tipo_servicio=null."
        )

        bloque_historial = ""
        if historial_reciente:
            lineas_hist = []
            for item in historial_reciente[-6:]:
                rol = "Cliente" if item.get("rol") == "cliente" else "Asesor"
                txt = str(item.get("texto") or item.get("contenido") or "").strip()
                if txt:
                    lineas_hist.append(f"- {rol}: {txt[:260]}")
            if lineas_hist:
                bloque_historial = "Historial reciente:\n" + "\n".join(lineas_hist) + "\n\n"

        estado_resumen = {
            "contexto_evento": contexto_tipo,
            "campos_pendientes_cotizacion": campos_pendientes,
            "parametros_acumulados": {
                k: v
                for k, v in parametros_acumulados.items()
                if not str(k).startswith("_")
            },
            "tiene_cotizacion": bool((estado_conversacion or {}).get("tiene_cotizacion")),
            "ultima_cotizacion": (estado_conversacion or {}).get("ultima_cotizacion"),
            "cotizacion_aprobada": bool((estado_conversacion or {}).get("cotizacion_aprobada")),
            "paso_reserva": (estado_conversacion or {}).get("paso_reserva"),
            "fecha_actual_bogota": datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M"),
        }

        usuario = (
            f"Estado actual de la conversacion:\n{json.dumps(estado_resumen, ensure_ascii=False)}\n\n"
            f"{bloque_historial}"
            f"Ultimo mensaje del cliente:\n\"\"\"{texto}\"\"\""
        )
        return [
            {"role": "system", "content": sistema},
            {"role": "user", "content": usuario},
        ]

    def _invocar_llama_nlu(
        self,
        texto: str,
        contexto_tipo: Optional[str],
        campos_pendientes: List[str],
        parametros_acumulados: Dict[str, Any],
        historial_reciente: Optional[List[Dict[str, Any]]],
        estado_conversacion: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.activo or self.en_enfriamiento:
            return None
        mensajes = self._construir_mensajes_nlu(
            texto=texto,
            contexto_tipo=contexto_tipo,
            campos_pendientes=campos_pendientes,
            parametros_acumulados=parametros_acumulados,
            historial_reciente=historial_reciente,
            estado_conversacion=estado_conversacion,
        )
        try:
            resp = self._cliente.chat(
                model=self.modelo,
                messages=mensajes,
                format="json",
                options={"temperature": 0.0, "num_predict": 240, "num_ctx": _NUM_CTX_LLAMA},
                keep_alive="30m",
            )
            contenido = (
                getattr(getattr(resp, "message", None), "content", None)
                or (resp.get("message", {}).get("content") if isinstance(resp, dict) else "")
                or ""
            )
            datos = json.loads(contenido)
            if not isinstance(datos, dict):
                self.fallbacks_activados += 1
                return None
            self.llamadas_nlu_exitosas += 1
            self.ultimo_error = ""
            return datos
        except Exception as exc:
            self._activar_enfriamiento(20.0, f"Error NLU Ollama: {exc}")
            return None

    def _normalizar_analisis_llama(
        self,
        datos_llama: Dict[str, Any],
        texto: str,
        contexto_tipo: Optional[str],
        campos_pendientes: List[str],
        parametros_acumulados: Dict[str, Any],
        estado_conversacion: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Valida y normaliza la salida JSON de Llama aplicando guardas de negocio."""
        texto_norm = _normalizar_texto_espanol(texto)
        intencion = str(datos_llama.get("intencion") or "consulta_servicio")
        tono = str(datos_llama.get("tono_cliente") or "interesado")
        if tono not in {"interesado", "indeciso", "hostil"}:
            tono = "interesado"

        es_saludo = bool(datos_llama.get("es_saludo_simple"))
        es_pregunta_cat = bool(datos_llama.get("es_pregunta_catalogo")) or _es_pregunta_catalogo(texto_norm)
        cotizacion_explicita = bool(datos_llama.get("cotizacion_explicita")) or _es_solicitud_cotizacion_explicita(texto_norm)
        escalamiento_humano = bool(datos_llama.get("escalamiento_humano"))
        paso_reserva = str((estado_conversacion or {}).get("paso_reserva") or "")
        recolectando_datos_reserva = paso_reserva in {"nombre_cliente", "cedula", "telefono"}

        # Guarda determinista de seguridad para Human-in-the-Loop (defensa en profundidad)
        senal_reclamo = any(
            _contiene_termino(texto_norm, palabra) for palabra in _PALABRAS_HOSTILES_SEGURIDAD
        )
        if senal_reclamo or (
            contexto_tipo in {"reclamo_critico", "reclamo_critico_cliente"}
            and not campos_pendientes
        ):
            intencion = "reclamo_critico"
            tono = "hostil"
            escalamiento_humano = True
            informacion_no_disponible = False
        else:
            senal_sin_info_det = _detectar_consulta_sin_informacion(
                texto_norm,
                contexto_negocio=self.contexto_negocio,
                catalogo_servicios=self.catalogo_servicios,
            )
            senal_sin_info_llama = bool(datos_llama.get("informacion_no_disponible")) or (
                intencion == "consulta_sin_informacion"
            )
            # Evitar falsos positivos si solo entrega nombre/cedula/telefono en paso de reserva
            if recolectando_datos_reserva and "?" not in texto and not senal_sin_info_det:
                senal_sin_info_llama = False
                if intencion == "consulta_sin_informacion":
                    intencion = "consulta_servicio"
                    escalamiento_humano = False
            informacion_no_disponible = bool(
                (senal_sin_info_llama or senal_sin_info_det)
                and not (bool(datos_llama.get("fuera_de_alcance")) and cotizacion_explicita)
            )
            if informacion_no_disponible:
                intencion = "consulta_sin_informacion"
                escalamiento_humano = True
                es_saludo = False
                es_pregunta_cat = False
                cotizacion_explicita = False
                if tono == "hostil":
                    tono = "interesado"

        try:
            nivel_indecision = float(datos_llama.get("nivel_indecision", 0.20))
        except (TypeError, ValueError):
            nivel_indecision = 0.20
        nivel_indecision = max(0.0, min(1.0, nivel_indecision))

        params: Dict[str, Any] = {}
        tipo_raw = datos_llama.get("tipo_servicio")
        meta_servicio: Dict[str, Any] = {}
        if isinstance(tipo_raw, str) and tipo_raw in self.catalogo_servicios:
            params["tipo_servicio"] = tipo_raw
            meta_servicio = self.catalogo_servicios[tipo_raw]
        elif (
            parametros_acumulados.get("tipo_servicio") in self.catalogo_servicios
            and intencion == "parametros_servicio"
        ):
            meta_servicio = self.catalogo_servicios[
                str(parametros_acumulados["tipo_servicio"])
            ]

        # Si es saludo simple o pregunta de catalogo pura, nunca forzar servicio ni cotizacion
        if (es_saludo or es_pregunta_cat) and not cotizacion_explicita and not escalamiento_humano:
            intencion = "consulta_servicio"
            params.pop("tipo_servicio", None)
            meta_servicio = {}

        # Cantidad y unidad comercial (con respaldo explicito si Llama omitio 'una sesion' / 'N horas')
        unidad_cfg = str(meta_servicio.get("unidad_cantidad") or self.unidad_cantidad)
        cantidad_llama = self._entero_positivo(datos_llama.get("cantidad_unidades"))
        if cantidad_llama is None and not es_saludo and not es_pregunta_cat:
            variantes_u = _variantes_unidad_comercial(unidad_cfg)
            u_regex = "|".join(re.escape(u) for u in variantes_u if u)
            n_regex = "|".join(sorted(_NUMEROS_ES, key=len, reverse=True))
            m_cant = re.search(
                rf"(?<!\w)(?P<cantidad>\d+|{n_regex})(?:\s+(?:sol[oa]|unica|unico))?\s+(?P<unidad>{u_regex})(?!\w)",
                texto_norm,
            )
            if m_cant:
                cantidad_llama = _numero_espanol_a_entero(m_cant.group("cantidad"))
        if cantidad_llama is not None and not es_saludo and not es_pregunta_cat:
            etiqueta_cant = self._etiqueta_cantidad(meta_servicio, unidad_cfg)
            params["cantidad_unidades"] = cantidad_llama
            params["unidad_cantidad"] = unidad_cfg
            params[etiqueta_cant] = cantidad_llama
            if normalizar_unidad_comercial(unidad_cfg) == "hora":
                params["horas_estimadas"] = cantidad_llama
            elif normalizar_unidad_comercial(unidad_cfg) == "sesion":
                params["sesiones_requeridas"] = cantidad_llama

        # Complejidad validada dentro del rango configurado
        comp_llama = self._entero_no_negativo(datos_llama.get("complejidad"))
        minimo_c, maximo_c = rango_complejidad_servicio(meta_servicio, self.rango_complejidad)
        if comp_llama is not None and minimo_c <= comp_llama <= maximo_c:
            params["complejidad"] = comp_llama

        # Si el cliente respondio un unico numero para un campo pendiente
        if campos_pendientes and not escalamiento_humano and not es_pregunta_cat:
            resp_pend = self._extraer_respuesta_pendiente(
                texto_norm, campos_pendientes, parametros_acumulados
            )
            for k, v in resp_pend.items():
                params.setdefault(k, v)
            if resp_pend and intencion == "consulta_servicio":
                intencion = "parametros_servicio"

        urg_llama = self._entero_positivo(datos_llama.get("urgencia_dias"))
        if urg_llama is not None:
            params["urgencia_dias"] = urg_llama

        if bool(datos_llama.get("incluye_adicional")):
            params["incluye_adicional"] = 1
            params["incluye_mantenimiento"] = 1

        desc_llama = datos_llama.get("descuento_solicitado_cliente")
        if desc_llama is not None:
            try:
                desc_val = float(desc_llama)
                if 0 < desc_val <= 100:
                    params["descuento_solicitado_cliente"] = desc_val
            except (TypeError, ValueError):
                pass
        else:
            match_desc = re.search(r"(\d{1,2})\s*%", texto_norm)
            if match_desc:
                params["descuento_solicitado_cliente"] = float(match_desc.group(1))

        # Parametros de fotografia a medida / fraccionada
        if normalizar_unidad_comercial(self.unidad_cantidad) == "sesion":
            cant_fotos = self._entero_positivo(datos_llama.get("cantidad_fotos"))
            cant_personas = self._entero_positivo(datos_llama.get("cantidad_personas"))
            en_ext = datos_llama.get("en_exterior")
            if cant_fotos is not None:
                params["cantidad_fotos"] = cant_fotos
            if cant_personas is not None:
                params["cantidad_personas"] = cant_personas
            if isinstance(en_ext, bool):
                params["en_exterior"] = en_ext
            if bool(datos_llama.get("es_personalizado")) or (
                cant_fotos is not None or cant_personas is not None or isinstance(en_ext, bool)
            ):
                if intencion not in {"consulta_servicio", "reclamo_critico", "rechazo_compra", "objecion_precio"}:
                    params["es_personalizado"] = True
                    desc_p = datos_llama.get("descripcion_personalizada")
                    if isinstance(desc_p, str) and desc_p.strip():
                        params["descripcion_personalizada"] = desc_p.strip()
                    if "tipo_servicio" not in params:
                        clave_def = (
                            str(parametros_acumulados["tipo_servicio"])
                            if parametros_acumulados.get("tipo_servicio") in self.catalogo_servicios
                            else (
                                "retrato_corporativo"
                                if "retrato_corporativo" in self.catalogo_servicios
                                else next(iter(self.catalogo_servicios.keys()), "retrato_corporativo")
                            )
                        )
                        params["tipo_servicio"] = clave_def
                        meta_servicio = self.catalogo_servicios.get(clave_def, {})
                    if "descripcion_personalizada" not in params:
                        nombre_base = str(meta_servicio.get("nombre", "Fotografia"))
                        if cant_fotos == 1:
                            params["descripcion_personalizada"] = f"Fotografia Individual a Medida (1 foto - base {nombre_base})"
                        elif cant_fotos:
                            params["descripcion_personalizada"] = f"Paquete a Medida ({cant_fotos} fotos - base {nombre_base})"
                        else:
                            params["descripcion_personalizada"] = f"Servicio a Medida ({nombre_base})"
                    if "cantidad_unidades" not in params:
                        etiqueta_u = self._etiqueta_cantidad(meta_servicio, unidad_cfg)
                        params["cantidad_unidades"] = 1
                        params["unidad_cantidad"] = unidad_cfg
                        params[etiqueta_u] = 1

        if bool(datos_llama.get("fuera_de_alcance")) and "tipo_servicio" not in params:
            params["fuera_de_alcance"] = True

        # Parseo preciso de fecha y hora en zona America/Bogota
        fecha_referencia = (
            parametros_acumulados.get("fecha_hora")
            or parametros_acumulados.get("fecha_agendamiento_pendiente")
            or parametros_acumulados.get("fecha_agendamiento_fecha")
        )
        fecha_texto_llama = str(datos_llama.get("fecha_hora_texto") or "").strip()
        fecha_hora = _extraer_fecha_hora(texto_norm, fecha_referencia=fecha_referencia)
        if fecha_hora is None and fecha_texto_llama:
            fecha_hora = _extraer_fecha_hora(
                _normalizar_texto_espanol(fecha_texto_llama),
                fecha_referencia=fecha_referencia,
            )
        if fecha_hora is not None:
            referencia_completa = False
            if fecha_referencia:
                try:
                    ref_dt = datetime.fromisoformat(
                        str(fecha_referencia).replace("Z", "+00:00")
                    )
                    referencia_completa = ref_dt.hour != 0 or ref_dt.minute != 0
                except ValueError:
                    referencia_completa = False
            hora_explicita = (
                _extraer_hora(texto_norm) is not None
                or (bool(fecha_texto_llama) and _extraer_hora(fecha_texto_llama) is not None)
            )
            if (
                not hora_explicita
                and not referencia_completa
                and fecha_hora.hour == 0
                and fecha_hora.minute == 0
            ):
                params["fecha_agendamiento_fecha"] = fecha_hora.date().isoformat()
                params["fecha_hora_falta"] = True
            else:
                params["fecha_hora"] = fecha_hora.strftime("%Y-%m-%d %H:%M")
                params["fecha_hora_pasada"] = bool(
                    fecha_hora <= datetime.now(ZoneInfo("America/Bogota"))
                )

        if intencion == "objecion_precio":
            tono = "indeciso"
            desc_sol = float(params.get("descuento_solicitado_cliente", 0.0))
            if desc_sol >= 18.0 or any(
                w in texto_norm for w in ("muy grande", "demasiado", "mitad", "20%", "25%", "30%")
            ):
                nivel_indecision = max(nivel_indecision, 0.96)
            else:
                nivel_indecision = max(nivel_indecision, 0.72)
        elif intencion == "reclamo_critico":
            tono = "hostil"
            escalamiento_humano = True
            nivel_indecision = max(nivel_indecision, 0.90)
        elif intencion == "consulta_sin_informacion":
            escalamiento_humano = True
            informacion_no_disponible = True
            if tono == "hostil":
                tono = "interesado"

        motivo_escalamiento = (
            "informacion_no_disponible"
            if informacion_no_disponible
            else ("reclamo_critico" if escalamiento_humano else "")
        )

        return {
            "intencion": intencion,
            "es_saludo_simple": es_saludo,
            "es_pregunta_catalogo": es_pregunta_cat,
            "tono_cliente": tono,
            "nivel_indecision": round(nivel_indecision, 2),
            "confianza": 0.95,
            "motor_nlu": f"llama_local:{self.modelo}",
            "escalamiento_humano": escalamiento_humano,
            "informacion_no_disponible": bool(informacion_no_disponible),
            "motivo_escalamiento": motivo_escalamiento,
            "cotizacion_explicita": bool(cotizacion_explicita),
            "parametros_extraidos": params,
        }

    # ------------------------------------------------------------------
    # ANALIZADOR DETERMINISTA DE RESPALDO / PRUEBAS OFFLINE (SIN ML)
    # ------------------------------------------------------------------
    def _analizar_determinista_sin_ml(
        self,
        texto: str,
        contexto_tipo: Optional[str] = None,
        campos_pendientes: Optional[List[str]] = None,
        parametros_acumulados: Optional[Dict[str, Any]] = None,
        estado_conversacion: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Respaldo determinista puro (sin Scikit-Learn ni ML) utilizado unicamente
        en pruebas unitarias offline o si el demonio local Ollama no responde.
        """
        texto_norm = _normalizar_texto_espanol(texto)
        parametros_acumulados = parametros_acumulados or {}
        parametros_pendientes = self._extraer_respuesta_pendiente(
            texto_norm, campos_pendientes or [], parametros_acumulados
        )
        parametros = self._extraer_parametros_determinista(
            texto_norm, parametros_acumulados=parametros_acumulados
        )
        if parametros_pendientes:
            parametros.update(parametros_pendientes)

        objecion_precio_explicita = _es_objecion_precio_explicita(texto_norm)
        cotizacion_explicita = _es_solicitud_cotizacion_explicita(texto_norm)
        es_pregunta_cat = _es_pregunta_catalogo(texto_norm) and not cotizacion_explicita
        seleccion_paquete = _es_seleccion_paquete(texto_norm)
        es_saludo = bool(
            not es_pregunta_cat
            and not cotizacion_explicita
            and re.match(
                r"^(?:hola|holi|buenos?\s+dias?|buenas?(?:\s+(?:tardes|noches))?|buen\s+dia|hey|que\s+tal|saludos)\W*$",
                texto_norm.strip(),
            )
        )

        referencia_agenda = (
            parametros_acumulados.get("fecha_hora")
            or parametros_acumulados.get("fecha_agendamiento_pendiente")
            or parametros_acumulados.get("fecha_agendamiento_fecha")
        )
        seleccion_horario = bool(
            referencia_agenda
            and _extraer_hora(texto_norm)
            and not objecion_precio_explicita
        )

        senal_reclamo = any(
            _contiene_termino(texto_norm, palabra) for palabra in _PALABRAS_HOSTILES_SEGURIDAD
        )
        reclamo_explicito = senal_reclamo or (
            contexto_tipo in {"reclamo_critico_cliente", "reclamo_critico"}
            and not parametros_pendientes
        )
        paso_reserva = str((estado_conversacion or {}).get("paso_reserva") or "")
        recolectando_datos_reserva = paso_reserva in {"nombre_cliente", "cedula", "telefono"}
        consulta_sin_info = (
            not reclamo_explicito
            and not parametros_pendientes
            and not (recolectando_datos_reserva and "?" not in texto)
            and not (parametros.get("fuera_de_alcance") and cotizacion_explicita)
            and _detectar_consulta_sin_informacion(
                texto_norm,
                contexto_negocio=self.contexto_negocio,
                catalogo_servicios=self.catalogo_servicios,
            )
        )
        escalamiento_humano = False
        informacion_no_disponible = False

        if reclamo_explicito:
            intencion = "reclamo_critico"
            tono = "hostil"
            nivel_indecision = 0.90
            escalamiento_humano = True
        elif consulta_sin_info:
            intencion = "consulta_sin_informacion"
            tono = "interesado"
            nivel_indecision = 0.25
            escalamiento_humano = True
            informacion_no_disponible = True
            es_saludo = False
            es_pregunta_cat = False
            cotizacion_explicita = False
        elif es_pregunta_cat and not cotizacion_explicita:
            intencion = "consulta_servicio"
            tono = "interesado"
            nivel_indecision = 0.20
        elif parametros_pendientes:
            intencion = "solicitud_cotizacion"
            tono = "interesado"
            nivel_indecision = 0.20
        elif re.fullmatch(r"[+\d\s().-]+", texto.strip()):
            intencion = "consulta_servicio"
            tono = "interesado"
            nivel_indecision = 0.20
        elif contexto_tipo in {"rechazo_compra", "rechazo_compra_no_interesado"} or any(
            _contiene_termino(texto_norm, palabra) for palabra in _PALABRAS_RECHAZO
        ):
            intencion = "rechazo_compra"
            tono = "indeciso"
            nivel_indecision = 0.85
        elif objecion_precio_explicita:
            intencion = "objecion_precio"
            tono = "indeciso"
            desc_sol = float(parametros.get("descuento_solicitado_cliente", 0.0))
            if desc_sol >= 18.0 or any(
                w in texto_norm for w in ("muy grande", "demasiado", "mitad", "20%", "25%", "30%")
            ):
                nivel_indecision = 0.96
            else:
                nivel_indecision = 0.72
        elif (
            contexto_tipo in {"aceptar_cotizacion", "solicitud_agendamiento"}
            or seleccion_horario
            or _extraer_hora(texto_norm) is not None
            or "fecha_hora" in parametros
            or "fecha_agendamiento_fecha" in parametros
            or any(_contiene_termino(texto_norm, p) for p in _PALABRAS_AGENDA)
        ):
            intencion = "solicitud_agendamiento"
            tono = "interesado"
            nivel_indecision = 0.12
        elif (
            cotizacion_explicita
            or contexto_tipo == "parametros_servicio"
            or (
                not es_pregunta_cat
                and (
                    bool(parametros.get("es_personalizado"))
                    or (
                        parametros.get("cantidad_fotos") is not None
                        and normalizar_unidad_comercial(self.unidad_cantidad) == "sesion"
                    )
                    or seleccion_paquete
                    or any(
                        k in parametros
                        for k in (
                            "tipo_servicio",
                            "cantidad_unidades",
                            "horas_estimadas",
                            "sesiones_requeridas",
                            "complejidad",
                            "fuera_de_alcance",
                        )
                    )
                )
            )
            or contexto_tipo == "solicitud_cotizacion"
        ):
            intencion = "solicitud_cotizacion"
            tono = "interesado"
            nivel_indecision = 0.20
        else:
            intencion = "consulta_servicio"
            tono = "interesado"
            nivel_indecision = 0.20

        motivo_escalamiento = (
            "informacion_no_disponible"
            if informacion_no_disponible
            else ("reclamo_critico" if escalamiento_humano else "")
        )

        return {
            "intencion": intencion,
            "es_saludo_simple": es_saludo,
            "es_pregunta_catalogo": es_pregunta_cat,
            "tono_cliente": tono,
            "nivel_indecision": round(nivel_indecision, 2),
            "confianza": 1.0,
            "motor_nlu": "respaldo_determinista_sin_ml",
            "escalamiento_humano": escalamiento_humano,
            "informacion_no_disponible": informacion_no_disponible,
            "motivo_escalamiento": motivo_escalamiento,
            "cotizacion_explicita": bool(cotizacion_explicita),
            "parametros_extraidos": parametros,
        }

    def analizar_mensaje(
        self,
        texto: str,
        contexto_tipo: Optional[str] = None,
        campos_pendientes: Optional[List[str]] = None,
        parametros_acumulados: Optional[Dict[str, Any]] = None,
        historial_reciente: Optional[List[Dict[str, Any]]] = None,
        estado_conversacion: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analiza el mensaje del cliente mediante Llama local (Ollama Structured Outputs).
        Si Llama no esta habilitado (tests offline) o hay fallo de conexion temporal,
        recurre al analizador determinista sin ML para no bloquear los agentes SPADE.
        """
        pendientes = list(campos_pendientes or [])
        acumulados = dict(parametros_acumulados or {})
        datos_llama = self._invocar_llama_nlu(
            texto=texto,
            contexto_tipo=contexto_tipo,
            campos_pendientes=pendientes,
            parametros_acumulados=acumulados,
            historial_reciente=historial_reciente,
            estado_conversacion=estado_conversacion,
        )
        if datos_llama is not None:
            return self._normalizar_analisis_llama(
                datos_llama=datos_llama,
                texto=texto,
                contexto_tipo=contexto_tipo,
                campos_pendientes=pendientes,
                parametros_acumulados=acumulados,
                estado_conversacion=estado_conversacion,
            )
        return self._analizar_determinista_sin_ml(
            texto=texto,
            contexto_tipo=contexto_tipo,
            campos_pendientes=pendientes,
            parametros_acumulados=acumulados,
            estado_conversacion=estado_conversacion,
        )

    # ------------------------------------------------------------------
    # NLG CON LLAMA LOCAL Y VALIDADOR ANTI-ALUCINACION
    # ------------------------------------------------------------------
    @staticmethod
    def _validar_integridad(
        respuesta_llm: str,
        texto_base: str,
        hechos_obligatorios: Iterable[str],
    ) -> bool:
        """Verifica que Llama no invente precios ni altere montos o datos verificados."""
        limpia = (respuesta_llm or "").strip()
        if not limpia or len(limpia) < 12 or len(limpia) > 1500:
            return False

        montos_base = _extraer_montos_normalizados(texto_base)
        montos_llm = _extraer_montos_normalizados(limpia)
        if not montos_base and montos_llm:
            return False
        if montos_llm - montos_base:
            return False
        if montos_base and not montos_llm:
            return False

        norm_llm = _normalizar_comparacion(limpia)
        if any(
            termino in norm_llm
            for termino in (
                "fipa",
                "spade",
                "random forest",
                "knn",
                "ollama",
                "llama",
                "borrador",
                "no puedo proporcionar",
                "sin limite fijo",
                "fotos ilimitadas",
            )
        ):
            return False

        for hecho in hechos_obligatorios:
            hecho_limpio = str(hecho or "").strip()
            if not hecho_limpio:
                continue
            if _normalizar_comparacion(hecho_limpio) not in norm_llm:
                return False
        return True

    def redactar_respuesta(
        self,
        tipo_respuesta: str,
        texto_base: str,
        contexto_negocio: Optional[Dict[str, Any]] = None,
        mensaje_cliente: str = "",
        historial_reciente: Optional[List[Dict[str, Any]]] = None,
        hechos_obligatorios: Optional[Iterable[str]] = None,
    ) -> str:
        """Redacta una respuesta comercial natural con Llama local preservando hechos."""
        if not self.activo or self.en_enfriamiento:
            return texto_base

        hechos = list(hechos_obligatorios or [])
        cfg = contexto_negocio or self.contexto_negocio
        nombre_neg = str(cfg.get("nombre_negocio", "el estudio"))
        rubro = str(cfg.get("rubro", "servicios profesionales"))

        bloque_historial = ""
        if historial_reciente:
            lineas_hist = []
            for item in historial_reciente[-6:]:
                rol = "Cliente" if item.get("rol") == "cliente" else "Asesor"
                txt = str(item.get("texto") or item.get("contenido") or "").strip()
                if txt:
                    lineas_hist.append(f"- {rol}: {txt[:220]}")
            if lineas_hist:
                bloque_historial = "Historial reciente de la conversacion:\n" + "\n".join(lineas_hist) + "\n\n"

        prompt_usuario = (
            f"Negocio: {nombre_neg} ({rubro}).\n"
            f"Tipo de respuesta: {tipo_respuesta}.\n"
            f"{bloque_historial}"
            f"Ultimo mensaje del cliente: {mensaje_cliente or '(sin texto)'}\n"
            f"Borrador verificado (hechos obligatorios que debes respetar intactos):\n"
            f"\"\"\"{texto_base}\"\"\"\n\n"
            "Redacta la respuesta final para enviar al cliente:"
        )

        try:
            resp = self._cliente.chat(
                model=self.modelo,
                messages=[
                    {"role": "system", "content": _INSTRUCCION_SISTEMA_NLG},
                    {"role": "user", "content": prompt_usuario},
                ],
                options={"temperature": 0.45, "num_predict": 260, "num_ctx": _NUM_CTX_LLAMA},
                keep_alive="30m",
            )
            texto_generado = str(
                getattr(getattr(resp, "message", None), "content", None)
                or (resp.get("message", {}).get("content") if isinstance(resp, dict) else "")
                or ""
            ).strip()
            if texto_generado.startswith('"') and texto_generado.endswith('"'):
                texto_generado = texto_generado[1:-1].strip()
            if self._validar_integridad(texto_generado, texto_base, hechos):
                self.llamadas_nlg_exitosas += 1
                self.ultimo_error = ""
                return texto_generado
            self.fallbacks_activados += 1
            self.ultimo_error = "Validacion anti-alucinacion: respuesta conservo borrador base"
            return texto_base
        except Exception as exc:
            self._activar_enfriamiento(20.0, f"Error NLG Ollama: {exc}")
            return texto_base

    # ------------------------------------------------------------------
    # UTILIDADES DE VALIDACION Y EXTRACCION DETERMINISTA (SIN ML)
    # ------------------------------------------------------------------
    def campos_cotizacion_pendientes(
        self, parametros: Dict[str, Any]
    ) -> List[str]:
        """Calcula los datos de cotizacion que siguen faltando segun el servicio."""
        pendientes: List[str] = []
        tipo = parametros.get("tipo_servicio")
        meta = self.catalogo_servicios.get(str(tipo), {}) if tipo else {}
        if not tipo or str(tipo) not in self.catalogo_servicios:
            pendientes.append("tipo_servicio")

        unidad = str(meta.get("unidad_cantidad") or self.unidad_cantidad)
        etiqueta_cantidad = self._etiqueta_cantidad(meta, unidad)
        unidad_incompatible = bool(
            parametros.get("unidad_cantidad")
            and normalizar_unidad_comercial(parametros["unidad_cantidad"])
            != normalizar_unidad_comercial(unidad)
        )
        claves_cantidad = [etiqueta_cantidad, "cantidad_unidades"]
        if normalizar_unidad_comercial(unidad) == "hora":
            claves_cantidad.append("horas_estimadas")
        elif normalizar_unidad_comercial(unidad) == "sesion":
            claves_cantidad.append("sesiones_requeridas")
        cantidad = next(
            (
                self._entero_positivo(parametros.get(clave))
                for clave in dict.fromkeys(claves_cantidad)
                if self._entero_positivo(parametros.get(clave)) is not None
            ),
            None,
        )
        if unidad_incompatible or cantidad is None:
            pendientes.append(etiqueta_cantidad)

        requiere_complejidad = bool(meta.get("requiere_complejidad", True))
        if parametros.get("es_personalizado") or parametros.get("cantidad_fotos") is not None:
            requiere_complejidad = False
        complejidad = self._entero_no_negativo(parametros.get("complejidad"))
        minimo, maximo = rango_complejidad_servicio(meta, self.rango_complejidad)
        complejidad_valida = (
            complejidad is not None and minimo <= complejidad <= maximo
        )
        if requiere_complejidad and not complejidad_valida:
            pendientes.append(
                str(meta.get("etiqueta_complejidad", self.etiqueta_complejidad))
            )
        return pendientes

    def _etiqueta_cantidad(
        self, meta_servicio: Dict[str, Any], unidad: str
    ) -> str:
        if meta_servicio.get("etiqueta_cantidad"):
            return str(meta_servicio["etiqueta_cantidad"])
        if (
            self.etiqueta_cantidad
            and normalizar_unidad_comercial(unidad)
            == normalizar_unidad_comercial(self.unidad_cantidad)
        ):
            return self.etiqueta_cantidad
        return campo_cantidad_para_unidad(unidad)

    def _extraer_respuesta_pendiente(
        self,
        texto_norm: str,
        campos_pendientes: List[str],
        parametros_acumulados: Dict[str, Any],
    ) -> Dict[str, Any]:
        if len(campos_pendientes) != 1:
            return {}
        valor_texto = texto_norm.strip(" \t\r\n.,!?;:")
        valor = _numero_espanol_a_entero(valor_texto)
        if valor is None:
            return {}

        campo = campos_pendientes[0]
        tipo = parametros_acumulados.get("tipo_servicio")
        meta = self.catalogo_servicios.get(str(tipo), {}) if tipo else {}
        if campo in {
            "complejidad",
            str(meta.get("etiqueta_complejidad", self.etiqueta_complejidad)),
        }:
            minimo, maximo = rango_complejidad_servicio(
                meta, self.rango_complejidad
            )
            return {"complejidad": valor} if minimo <= valor <= maximo else {}
        if campo == "cantidad_productos":
            return {"cantidad_productos": valor}

        unidad = str(meta.get("unidad_cantidad") or self.unidad_cantidad)
        if campo not in {
            "cantidad_unidades",
            self._etiqueta_cantidad(meta, unidad),
            campo_cantidad_para_unidad(unidad),
            "horas_estimadas",
            "sesiones_requeridas",
        } or valor <= 0:
            return {}
        resultado: Dict[str, Any] = {
            "cantidad_unidades": valor,
            "unidad_cantidad": unidad,
            self._etiqueta_cantidad(meta, unidad): valor,
        }
        if normalizar_unidad_comercial(unidad) == "hora":
            resultado["horas_estimadas"] = valor
        return resultado

    @staticmethod
    def _entero_positivo(valor: Any) -> Optional[int]:
        if valor is None or isinstance(valor, bool):
            return None
        try:
            numero = int(valor)
        except (TypeError, ValueError):
            return None
        return numero if numero > 0 else None

    @staticmethod
    def _entero_no_negativo(valor: Any) -> Optional[int]:
        if valor is None or isinstance(valor, bool):
            return None
        try:
            numero = int(valor)
        except (TypeError, ValueError):
            return None
        return numero if numero >= 0 else None

    def _extraer_parametros_determinista(
        self,
        texto_norm: str,
        parametros_acumulados: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extractor determinista ligero para modo de pruebas offline sin Ollama."""
        params: Dict[str, Any] = {}
        texto_norm = _normalizar_texto_espanol(texto_norm)
        meta_servicio_seleccionado: Dict[str, Any] = {}
        es_pregunta_cat = _es_pregunta_catalogo(texto_norm) and not _es_solicitud_cotizacion_explicita(texto_norm)

        if not es_pregunta_cat:
            for clave_servicio, meta in self.catalogo_servicios.items():
                clave_norm = _normalizar_texto_espanol(str(clave_servicio))
                nombre_norm = _normalizar_texto_espanol(str(meta.get("nombre", "")))
                if _contiene_termino(texto_norm, clave_norm) or _contiene_termino(
                    texto_norm, clave_norm.replace("_", " ")
                ):
                    params["tipo_servicio"] = clave_servicio
                    meta_servicio_seleccionado = meta
                    break
                palabras = [
                    _normalizar_texto_espanol(str(p))
                    for p in meta.get("palabras_clave", [])
                ]
                if any(_contiene_termino(texto_norm, palabra) for palabra in palabras):
                    params["tipo_servicio"] = clave_servicio
                    meta_servicio_seleccionado = meta
                    break
                if nombre_norm and any(
                    len(tok) >= 4 and _contiene_termino(texto_norm, tok)
                    for tok in nombre_norm.split()
                ):
                    params["tipo_servicio"] = clave_servicio
                    meta_servicio_seleccionado = meta
                    break

            if "tipo_servicio" not in params and self.catalogo_servicios:
                ordinales_catalogo = (
                    (r"\b(?:el\s+|la\s+)?primer[oa]\b|\b(?:paquete|opcion|numero)\s*(?:#\s*)?1\b", 0),
                    (r"\b(?:el\s+|la\s+)?segund[oa]\b|\b(?:paquete|opcion|numero)\s*(?:#\s*)?2\b", 1),
                    (r"\b(?:el\s+|la\s+)?tercer[oa]\b|\b(?:paquete|opcion|numero)\s*(?:#\s*)?3\b", 2),
                    (r"\b(?:el\s+|la\s+)?cuart[oa]\b|\b(?:paquete|opcion|numero)\s*(?:#\s*)?4\b", 3),
                    (r"\b(?:el\s+|la\s+)?quint[oa]\b|\b(?:paquete|opcion|numero)\s*(?:#\s*)?5\b", 4),
                )
                claves_ordenadas = list(self.catalogo_servicios.keys())
                for patron_ord, indice_ord in ordinales_catalogo:
                    if indice_ord < len(claves_ordenadas) and re.search(patron_ord, texto_norm):
                        clave_ord = claves_ordenadas[indice_ord]
                        params["tipo_servicio"] = clave_ord
                        meta_servicio_seleccionado = self.catalogo_servicios[clave_ord]
                        break

        unidad_configurada = str(
            meta_servicio_seleccionado.get("unidad_cantidad") or self.unidad_cantidad
        )
        variantes_unidad = _variantes_unidad_comercial(unidad_configurada)
        unidades_regex = "|".join(re.escape(u) for u in variantes_unidad if u)
        numeros_regex = "|".join(sorted(_NUMEROS_ES, key=len, reverse=True))
        match_cant = re.search(
            rf"(?<!\w)(?P<cantidad>\d+|{numeros_regex})"
            rf"(?:\s+(?:sol[oa]|unica|unico))?\s+"
            rf"(?P<unidad>{unidades_regex})(?!\w)",
            texto_norm,
        )
        if match_cant:
            cantidad = _numero_espanol_a_entero(match_cant.group("cantidad"))
            if cantidad and cantidad > 0:
                campo_cantidad = self._etiqueta_cantidad(
                    meta_servicio_seleccionado, unidad_configurada
                )
                params["cantidad_unidades"] = cantidad
                params["unidad_cantidad"] = unidad_configurada
                params[campo_cantidad] = cantidad
                if normalizar_unidad_comercial(unidad_configurada) == "hora":
                    params["horas_estimadas"] = cantidad
                elif normalizar_unidad_comercial(unidad_configurada) == "sesion":
                    params["sesiones_requeridas"] = cantidad
        else:
            match_otra_unidad = re.search(
                rf"(?<!\w)(?P<cantidad>\d+|{numeros_regex})"
                r"\s+(?P<unidad>horas?|hrs?|h|sesiones?|unidades?|dias?|fotografias?|fotos?|paginas?|piezas?|modulos?|proyectos?)(?!\w)",
                texto_norm,
            )
            if match_otra_unidad:
                cantidad = _numero_espanol_a_entero(match_otra_unidad.group("cantidad"))
                unidad_mencionada = match_otra_unidad.group("unidad")
                if cantidad and cantidad > 0:
                    unidad_canon = normalizar_unidad_comercial(unidad_mencionada)
                    if unidad_canon == "fotografia":
                        params["cantidad_fotos"] = cantidad

        if normalizar_unidad_comercial(self.unidad_cantidad) == "sesion":
            match_prod = re.search(
                rf"(?<!\w)(?P<prod>\d+|{numeros_regex})\s+productos?(?!\w)",
                texto_norm,
            )
            if match_prod:
                cant_prod = _numero_espanol_a_entero(match_prod.group("prod"))
                if cant_prod and cant_prod > 0:
                    params["cantidad_productos"] = cant_prod

            match_fotos = re.search(
                rf"(?<!\w)(?P<cantidad>\d+|{numeros_regex})"
                r"(?:\s+(?:sol[oa]|unica|unico))?\s+(?:fotografias?|fotos?)(?!\w)",
                texto_norm,
            )
            if match_fotos:
                cant_f = _numero_espanol_a_entero(match_fotos.group("cantidad"))
                if cant_f and cant_f > 0:
                    params["cantidad_fotos"] = cant_f

            match_pers = re.search(
                rf"(?:somos|para|de|con)\s+(?P<pers>\d+|{numeros_regex})\s+"
                r"(?:personas?|integrantes?|miembros?|familiares?|amigos?|adultos?|ninos?|empleados?)\b"
                rf"|\b(?P<pers2>\d+|{numeros_regex})\s+(?:personas?|integrantes?|miembros?)\b",
                texto_norm,
            )
            if match_pers:
                raw_p = match_pers.group("pers") or match_pers.group("pers2")
                cant_p = _numero_espanol_a_entero(raw_p)
                if cant_p and cant_p > 0:
                    params["cantidad_personas"] = cant_p
            elif re.search(r"\bcon\s+mi\s+(?:esposa|esposo|pareja|novia|novio)\b", texto_norm):
                params["cantidad_personas"] = 2

            if re.search(
                r"\b(?:exterior|exteriores|aire\s+libre|domicilio|locacion|parque|playa|finca|oficinas?|centro|iglesia|salon|afuera)\b",
                texto_norm,
            ):
                params["en_exterior"] = True
            elif re.search(r"\b(?:en\s+estudio|en\s+el\s+estudio|en\s+su\s+estudio)\b", texto_norm):
                params["en_exterior"] = False

            if "tipo_servicio" not in params:
                patrones_custom = (
                    (r"\b(?:familiar|familia|padres|hijos|hermanos)\b", "retrato_corporativo", "Fotografia Familiar a Medida", False),
                    (r"\b(?:pareja|novios|compromiso|preboda|aniversario|esposa|esposo)\b", "retrato_corporativo", "Sesion de Pareja a Medida", False),
                    (r"\b(?:maternidad|embarazo|embarazada|bebe|newborn|recien\s+nacido)\b", "retrato_corporativo", "Fotografia de Maternidad / Bebe", False),
                    (r"\b(?:boda|matrimonio|quince\s+anos|15\s+anos|bautizo|grado|graduacion|cumpleanos|fiesta)\b", "cobertura_evento", "Cobertura Fotografica de Evento Social", True),
                    (r"\b(?:mascota|mascotas|perro|perrito|gato|gatito)\b", "retrato_corporativo", "Sesion Fotografica con Mascota", False),
                    (r"\b(?:hoja\s+de\s+vida|cv|linkedin|perfil\s+profesional|foto\s+personal)\b", "retrato_corporativo", "Retrato Personal / Profesional", False),
                    (r"\b(?:platos|comida|menu|restaurante|gastronomia|alimentos)\b", "sesion_producto", "Fotografia Gastronomica a Medida", False),
                    (r"\b(?:apartamento|casa|airbnb|inmueble|arquitectura|propiedad)\b", "sesion_producto", "Fotografia Inmobiliaria / Arquitectura", True),
                    (r"\b(?:moda|lookbook|ropa|prendas|catalogo\s+de\s+ropa)\b", "sesion_producto", "Fotografia Editorial / Moda", False),
                )
                for patron_c, clave_ref, nombre_custom, default_ext in patrones_custom:
                    if re.search(patron_c, texto_norm):
                        clave_elegida = (
                            clave_ref
                            if clave_ref in self.catalogo_servicios
                            else next(iter(self.catalogo_servicios.keys()), clave_ref)
                        )
                        params["tipo_servicio"] = clave_elegida
                        meta_servicio_seleccionado = self.catalogo_servicios.get(clave_elegida, {})
                        params["es_personalizado"] = True
                        params["descripcion_personalizada"] = nombre_custom
                        if "en_exterior" not in params and default_ext:
                            params["en_exterior"] = True
                        break

            acum = parametros_acumulados or {}
            if not params.get("es_personalizado") and (
                "cantidad_personas" in params
                or "en_exterior" in params
                or "cantidad_fotos" in params
            ):
                params["es_personalizado"] = True
                if "tipo_servicio" not in params and acum.get("tipo_servicio"):
                    params["tipo_servicio"] = acum["tipo_servicio"]
                    meta_servicio_seleccionado = self.catalogo_servicios.get(
                        str(params["tipo_servicio"]), {}
                    )
                if "descripcion_personalizada" not in params and acum.get("descripcion_personalizada"):
                    params["descripcion_personalizada"] = acum["descripcion_personalizada"]

            if params.get("es_personalizado"):
                if "tipo_servicio" not in params:
                    clave_default = (
                        str(acum["tipo_servicio"])
                        if acum.get("tipo_servicio") in self.catalogo_servicios
                        else (
                            "retrato_corporativo"
                            if "retrato_corporativo" in self.catalogo_servicios
                            else next(iter(self.catalogo_servicios.keys()), "retrato_corporativo")
                        )
                    )
                    params["tipo_servicio"] = clave_default
                    meta_servicio_seleccionado = self.catalogo_servicios.get(clave_default, {})
                if "descripcion_personalizada" not in params:
                    nombre_base = str(meta_servicio_seleccionado.get("nombre", "Fotografia"))
                    cant_f = params.get("cantidad_fotos")
                    if cant_f == 1:
                        params["descripcion_personalizada"] = f"Fotografia Individual a Medida (1 foto - base {nombre_base})"
                    elif cant_f:
                        params["descripcion_personalizada"] = f"Paquete a Medida ({cant_f} fotos - base {nombre_base})"
                    else:
                        params["descripcion_personalizada"] = f"Servicio a Medida ({nombre_base})"
                if "cantidad_unidades" not in params:
                    campo_u = self._etiqueta_cantidad(meta_servicio_seleccionado, unidad_configurada)
                    params["cantidad_unidades"] = 1
                    params["unidad_cantidad"] = unidad_configurada
                    params[campo_u] = 1

        match_comp = re.search(
            rf"\b(?:nivel\s+de\s+)?complejidad(?:\s+de\s+nivel)?\s*(?:es\s+|:\s*)?(?P<valor>\d+|{numeros_regex})\b",
            texto_norm,
        ) or re.search(rf"\bnivel\s+(?P<valor>\d+|{numeros_regex})\b", texto_norm)
        if match_comp:
            complejidad = _numero_espanol_a_entero(match_comp.group("valor"))
            minimo, maximo = rango_complejidad_servicio(
                meta_servicio_seleccionado, self.rango_complejidad
            )
            if complejidad is not None and minimo <= complejidad <= maximo:
                params["complejidad"] = complejidad
        else:
            equivalencias = meta_servicio_seleccionado.get(
                "equivalencias_complejidad", self.equivalencias_complejidad
            )
            if isinstance(equivalencias, dict):
                for etiqueta, valor in equivalencias.items():
                    etiqueta_norm = _normalizar_texto_espanol(str(etiqueta))
                    if etiqueta_norm and re.search(
                        rf"\b(?:complejidad\s+)?{re.escape(etiqueta_norm)}\b",
                        texto_norm,
                    ):
                        complejidad = int(valor)
                        minimo, maximo = rango_complejidad_servicio(
                            meta_servicio_seleccionado, self.rango_complejidad
                        )
                        if minimo <= complejidad <= maximo:
                            params["complejidad"] = complejidad
                        break

        match_urg = re.search(r"(?:urgencia|en|para)\s*(\d+)\s*d[ií]as?", texto_norm)
        if match_urg:
            params["urgencia_dias"] = int(match_urg.group(1))

        if any(w in texto_norm for w in ("mantenimiento", "soporte", "edicion", "adicional")):
            params["incluye_adicional"] = 1
            params["incluye_mantenimiento"] = 1

        match_desc = re.search(r"(\d{1,2})\s*%", texto_norm)
        if match_desc:
            params["descuento_solicitado_cliente"] = float(match_desc.group(1))

        parametros_acumulados = parametros_acumulados or {}
        fecha_referencia = (
            parametros_acumulados.get("fecha_hora")
            or parametros_acumulados.get("fecha_agendamiento_pendiente")
            or parametros_acumulados.get("fecha_agendamiento_fecha")
        )
        fecha_hora = _extraer_fecha_hora(texto_norm, fecha_referencia=fecha_referencia)
        if fecha_hora:
            referencia_completa = False
            if fecha_referencia:
                try:
                    ref_dt = datetime.fromisoformat(
                        str(fecha_referencia).replace("Z", "+00:00")
                    )
                    referencia_completa = ref_dt.hour != 0 or ref_dt.minute != 0
                except ValueError:
                    referencia_completa = False
            hora_explicita = _extraer_hora(texto_norm) is not None
            if (
                not hora_explicita
                and not referencia_completa
                and fecha_hora.hour == 0
                and fecha_hora.minute == 0
            ):
                params["fecha_agendamiento_fecha"] = fecha_hora.date().isoformat()
                params["fecha_hora_falta"] = True
            else:
                params["fecha_hora"] = fecha_hora.strftime("%Y-%m-%d %H:%M")
                params["fecha_hora_pasada"] = bool(
                    fecha_hora <= datetime.now(ZoneInfo("America/Bogota"))
                )

        rubros_ajenos = (
            "plomeria", "carpinteria", "construccion", "mecanica", "cirugia",
            "odontologia", "peluqueria", "barberia", "zapatos", "zapatillas",
            "llantas", "repuestos", "medicamentos", "farmacia", "mudanza",
            "fumigacion", "cerrajeria", "electricista", "lavanderia",
            "pizza", "hamburguesa", "tiquetes aereos", "prestamo bancario",
            "clases de ingles", "reparacion de celulares", "arreglan celulares",
            "arreglar carros", "alquiler de carros",
        )
        if "tipo_servicio" not in params and any(
            _contiene_termino(texto_norm, rubro) for rubro in rubros_ajenos
        ):
            params["fuera_de_alcance"] = True

        return params
