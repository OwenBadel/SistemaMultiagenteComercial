"""
Repositorios de persistencia para EntornoVirtualAtencion:
- RepositorioSupabase: conexion en la nube via API REST HTTPS de Supabase.
- RepositorioSQLite: almacenamiento local SQLite (ideal para desarrollo offline y tests).

Ningun agente contiene datos fijos de negocio: la configuracion del independiente,
catalogo de servicios, historicos ML, agenda, perfiles y alertas viven en BD.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from src.ml.datasets import (
    CATALOGO_SERVICIOS,
    generar_historico_ofertas,
    generar_tarifas_historicas,
)


class RepositorioEntorno(ABC):
    """Contrato comun para lectura/escritura del entorno en BD nube o local."""

    modo_bd: str = "desconocido"

    def __init__(self, negocio_id: str) -> None:
        self.negocio_id = negocio_id

    @abstractmethod
    def obtener_configuracion_negocio(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def obtener_catalogo_servicios(self) -> Dict[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def obtener_promociones_vigentes(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def obtener_tarifas_historicas(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def obtener_historico_ofertas(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def listar_perfiles_clientes(self) -> Dict[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def guardar_perfil_cliente(self, perfil: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def listar_alertas_humanas(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def registrar_alerta_humana(self, alerta: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def resolver_alertas_cliente(self, id_cliente: str) -> None:
        ...

    @abstractmethod
    def listar_agendamientos_activos(
        self, fecha_inicio: str, fecha_fin: str
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def registrar_agendamiento(self, agendamiento: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def actualizar_agendamiento(
        self, id_agendamiento: str, cambios: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def obtener_agendamiento(
        self, id_agendamiento: str
    ) -> Optional[Dict[str, Any]]:
        ...


class ConflictoAgendamientoError(ValueError):
    """Señala que una cita activa ya ocupa parte del intervalo solicitado."""


def _fecha_utc(valor: str) -> datetime:
    fecha = datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    if fecha.tzinfo is None:
        raise ValueError("Las fechas de agendamiento deben incluir zona horaria.")
    return fecha.astimezone(timezone.utc)


def _fecha_bogota(valor: str) -> datetime:
    return _fecha_utc(valor).astimezone(ZoneInfo("America/Bogota"))


def _mapear_agendamiento(fila: Dict[str, Any]) -> Dict[str, Any]:
    resultado = dict(fila)
    for campo in ("fecha_inicio", "fecha_fin"):
        if resultado.get(campo):
            resultado[campo] = _fecha_bogota(str(resultado[campo])).isoformat()
    if isinstance(resultado.get("metadata"), str):
        resultado["metadata"] = _parse_json_o_lista(resultado["metadata"], {})
    return resultado


def _es_conflicto_agendamiento_supabase(respuesta: httpx.Response) -> bool:
    try:
        cuerpo = respuesta.json()
    except ValueError:
        cuerpo = {}
    if not isinstance(cuerpo, dict):
        cuerpo = {}

    codigo = str(cuerpo.get("code", ""))
    if codigo == "23P01":
        return True
    if respuesta.status_code != 409:
        return False

    detalle = " ".join(
        str(cuerpo.get(campo, ""))
        for campo in ("message", "details", "hint", "constraint")
    ).lower()
    if "agendamientos_sin_cruce_activo" in detalle or "exclusion" in detalle:
        return True
    return not codigo


def _parse_json_o_lista(valor: Any, fallback: Any) -> Any:
    if valor is None:
        return fallback
    if isinstance(valor, (list, dict)):
        return valor
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return fallback
        if texto.startswith(("[", "{")):
            try:
                return json.loads(texto)
            except json.JSONDecodeError:
                pass
        if isinstance(fallback, list):
            return [p.strip().lower() for p in texto.split(",") if p.strip()]
    return fallback


def _mapear_catalogo(filas: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    catalogo: Dict[str, Dict[str, Any]] = {}
    for row in filas:
        r = dict(row)
        tarifa = float(r["tarifa_unitaria"])
        palabras = [
            str(p).strip().lower()
            for p in _parse_json_o_lista(r.get("palabras_clave"), [])
            if str(p).strip()
        ]
        catalogo[str(r["clave"])] = {
            "id": int(r["servicio_id_num"]),
            "nombre": str(r["nombre"]),
            "descripcion": str(r["descripcion"]),
            "tarifa_unitaria": tarifa,
            "tarifa_hora_cop": tarifa,
            "palabras_clave": palabras,
            "requiere_complejidad": bool(r.get("requiere_complejidad", True)),
            "costo_adicional_mantenimiento": float(
                r.get("costo_adicional_mantenimiento", 380_000.0)
            ),
        }
    return catalogo


def _mapear_promociones(filas: Iterable[Any]) -> List[Dict[str, Any]]:
    promociones = []
    for row in filas:
        r = dict(row)
        promocion = {
            "codigo": str(r["codigo"]),
            "descripcion": str(r["descripcion"]),
            "descuento_max": float(r["descuento_max"]),
            "activa": bool(r.get("activa", True)),
        }
        condiciones = r.get("condiciones")
        if condiciones is not None:
            promocion["condiciones"] = _parse_json_o_lista(condiciones, {})
        promociones.append(promocion)
    return promociones


def _mapear_tarifas_df(filas: Iterable[Any]) -> pd.DataFrame:
    registros = []
    for row in filas:
        r = dict(row)
        cant = int(r["cantidad_unidades"])
        inc = int(r["incluye_adicional"])
        base = float(r["costo_base"])
        total = float(r["costo_total"])
        registros.append(
            {
                "id_cotizacion": r["id_cotizacion"],
                "tipo_servicio": r["tipo_servicio"],
                "tipo_servicio_id": int(r["tipo_servicio_id"]),
                "complejidad": int(r["complejidad"]),
                "cantidad_unidades": cant,
                "horas_estimadas": cant,
                "urgencia_dias": int(r["urgencia_dias"]),
                "incluye_adicional": inc,
                "incluye_mantenimiento": inc,
                "costo_base": base,
                "costo_base_cop": base,
                "costo_total": total,
                "costo_total_cop": total,
                "margen_estimado": float(r["margen_estimado"]),
            }
        )
    return pd.DataFrame(registros)


def _mapear_perfiles(filas: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    resultado: Dict[str, Dict[str, Any]] = {}
    for row in filas:
        r = dict(row)
        resultado[str(r["id_cliente"])] = {
            "id_cliente": str(r["id_cliente"]),
            "nombre": str(r["nombre"]),
            "canal": str(r["canal"]),
            "historial_interacciones": _parse_json_o_lista(
                r.get("historial_interacciones"), []
            ),
            "tono_actual": str(r.get("tono_actual", "neutral")),
            "fidelidad": int(r.get("fidelidad", 1)),
            "ultima_cotizacion": r.get("ultima_cotizacion"),
            "ultima_complejidad": int(r.get("ultima_complejidad", 3)),
            "parametros_parciales": _parse_json_o_lista(
                r.get("parametros_parciales"), {}
            ),
            "esperando_parametros_cotizacion": bool(
                r.get("esperando_parametros_cotizacion", False)
            ),
            "promocion_aplicada": r.get("promocion_aplicada"),
            "automatizacion_activa": bool(r.get("automatizacion_activa", True)),
        }
    return resultado


def _mapear_alertas(filas: Iterable[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": str(r["timestamp"]),
            "id_cliente": str(r["id_cliente"]),
            "nombre": str(r["nombre"]),
            "canal": str(r["canal"]),
            "motivo_critico": str(r["motivo_critico"]),
            "historial_charla": _parse_json_o_lista(
                dict(r).get("historial_charla"), []
            ),
            "estado": str(r["estado"]),
            "resuelta": bool(dict(r).get("resuelta", False)),
        }
        for r in filas
    ]


def _config_negocio_default(negocio_id: str) -> Dict[str, Any]:
    return {
        "id": negocio_id,
        "nombre_negocio": "Servicios Profesionales Independientes",
        "rubro": "Servicios Digitales",
        "moneda": "COP",
        "unidad_cantidad": "horas",
        "etiqueta_cantidad": "horas_estimadas",
        "etiqueta_complejidad": "complejidad",
        "telefono_dueno": "",
        "margen_base": 0.35,
        "margen_minimo": 0.22,
        "descuento_maximo": 15.0,
        "mensaje_catalogo_intro": "Con gusto te comparto nuestro portafolio de servicios disponibles",
    }


def _promos_default() -> List[Dict[str, Any]]:
    servicios_tech = list(CATALOGO_SERVICIOS.keys())
    return [
        {
            "codigo": "PROMO_CIERRE_10",
            "descripcion": "Incentivo de cierre rapido para clientes indecisos",
            "descuento_max": 12.0,
            "activa": True,
            "condiciones": {
                "servicios": servicios_tech,
                "cantidad_minima": 1,
                "cantidad_maxima": 500,
                "complejidad_minima": 1,
                "complejidad_maxima": 5,
            },
        },
        {
            "codigo": "PROMO_FIDELIDAD_15",
            "descripcion": "Beneficio tope de retencion comercial",
            "descuento_max": 15.0,
            "activa": True,
            "condiciones": {
                "servicios": servicios_tech,
                "cantidad_minima": 20,
                "cantidad_maxima": 500,
                "complejidad_minima": 1,
                "complejidad_maxima": 5,
            },
        },
    ]


class RepositorioSQLite(RepositorioEntorno):
    """Repositorio local SQLite con el mismo esquema relacional de Supabase."""

    modo_bd = "SQLite Local"

    def __init__(
        self,
        ruta_db: str | Path = ":memory:",
        negocio_id: str = "tech_freelance",
        auto_seed: bool = True,
    ) -> None:
        super().__init__(negocio_id=negocio_id)
        self.ruta_db = str(ruta_db)
        if self.ruta_db != ":memory:":
            Path(self.ruta_db).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.ruta_db, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._crear_tablas()
        if auto_seed:
            self._asegurar_datos_iniciales()

    def _crear_tablas(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS negocios (
                id TEXT PRIMARY KEY,
                nombre_negocio TEXT NOT NULL,
                rubro TEXT NOT NULL,
                moneda TEXT NOT NULL DEFAULT 'COP',
                unidad_cantidad TEXT NOT NULL DEFAULT 'horas',
                etiqueta_cantidad TEXT NOT NULL DEFAULT 'horas_estimadas',
                etiqueta_complejidad TEXT NOT NULL DEFAULT 'complejidad',
                telefono_dueno TEXT DEFAULT '',
                margen_base REAL NOT NULL DEFAULT 0.35,
                margen_minimo REAL NOT NULL DEFAULT 0.22,
                descuento_maximo REAL NOT NULL DEFAULT 15.0,
                mensaje_catalogo_intro TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS servicios (
                negocio_id TEXT NOT NULL,
                clave TEXT NOT NULL,
                servicio_id_num INTEGER NOT NULL,
                nombre TEXT NOT NULL,
                descripcion TEXT NOT NULL,
                tarifa_unitaria REAL NOT NULL,
                palabras_clave TEXT NOT NULL,
                requiere_complejidad INTEGER NOT NULL DEFAULT 1,
                costo_adicional_mantenimiento REAL NOT NULL DEFAULT 380000.0,
                PRIMARY KEY (negocio_id, clave)
            );
            CREATE TABLE IF NOT EXISTS promociones_vigentes (
                negocio_id TEXT NOT NULL,
                codigo TEXT NOT NULL,
                descripcion TEXT NOT NULL,
                descuento_max REAL NOT NULL,
                activa INTEGER NOT NULL DEFAULT 1,
                condiciones TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (negocio_id, codigo)
            );
            CREATE TABLE IF NOT EXISTS tarifas_historicas (
                negocio_id TEXT NOT NULL,
                id_cotizacion TEXT NOT NULL,
                tipo_servicio TEXT NOT NULL,
                tipo_servicio_id INTEGER NOT NULL,
                complejidad INTEGER NOT NULL,
                cantidad_unidades INTEGER NOT NULL,
                urgencia_dias INTEGER NOT NULL,
                incluye_adicional INTEGER NOT NULL,
                costo_base REAL NOT NULL,
                costo_total REAL NOT NULL,
                margen_estimado REAL NOT NULL,
                PRIMARY KEY (negocio_id, id_cotizacion)
            );
            CREATE TABLE IF NOT EXISTS historico_ofertas (
                negocio_id TEXT NOT NULL,
                id_oferta TEXT NOT NULL,
                monto_cotizado REAL NOT NULL,
                nivel_indecision REAL NOT NULL,
                complejidad INTEGER NOT NULL,
                fidelidad_cliente INTEGER NOT NULL,
                porcentaje_descuento_optimo REAL NOT NULL,
                cerrado_exitosamente INTEGER NOT NULL,
                PRIMARY KEY (negocio_id, id_oferta)
            );
            CREATE TABLE IF NOT EXISTS perfiles_clientes (
                negocio_id TEXT NOT NULL,
                id_cliente TEXT NOT NULL,
                nombre TEXT NOT NULL,
                canal TEXT NOT NULL,
                historial_interacciones TEXT NOT NULL,
                tono_actual TEXT NOT NULL,
                fidelidad INTEGER NOT NULL,
                ultima_cotizacion REAL,
                ultima_complejidad INTEGER NOT NULL DEFAULT 3,
                parametros_parciales TEXT NOT NULL,
                esperando_parametros_cotizacion INTEGER NOT NULL DEFAULT 0,
                promocion_aplicada REAL,
                automatizacion_activa INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (negocio_id, id_cliente)
            );
            CREATE TABLE IF NOT EXISTS alertas_humanas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                negocio_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                id_cliente TEXT NOT NULL,
                nombre TEXT NOT NULL,
                canal TEXT NOT NULL,
                motivo_critico TEXT NOT NULL,
                historial_charla TEXT NOT NULL,
                estado TEXT NOT NULL,
                resuelta INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS agendamientos (
                id TEXT PRIMARY KEY,
                negocio_id TEXT NOT NULL REFERENCES negocios(id) ON DELETE CASCADE,
                id_cliente TEXT NOT NULL,
                nombre_cliente TEXT NOT NULL,
                canal TEXT NOT NULL,
                servicio TEXT,
                fecha_inicio TEXT NOT NULL,
                fecha_fin TEXT NOT NULL,
                estado TEXT NOT NULL DEFAULT 'confirmada'
                    CHECK (estado IN ('pendiente', 'confirmada', 'cancelada', 'error')),
                registrado_en TEXT NOT NULL
                    CHECK (registrado_en IN ('supabase_y_google_calendar', 'solo_supabase')),
                google_calendar_estado TEXT NOT NULL
                    CHECK (google_calendar_estado IN ('pendiente', 'sincronizado', 'no_configurado', 'error')),
                google_event_id TEXT,
                google_event_link TEXT,
                google_calendar_error TEXT,
                observaciones TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (fecha_fin > fecha_inicio)
            );
            CREATE INDEX IF NOT EXISTS idx_agendamientos_negocio_fecha
                ON agendamientos (negocio_id, fecha_inicio, fecha_fin);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_agendamientos_google_event
                ON agendamientos (google_event_id) WHERE google_event_id IS NOT NULL;
            """
        )
        cols_promo = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(promociones_vigentes)").fetchall()
        }
        if "condiciones" not in cols_promo:
            self._conn.execute(
                "ALTER TABLE promociones_vigentes ADD COLUMN condiciones TEXT NOT NULL DEFAULT '{}'"
            )
        self._conn.commit()

    def _asegurar_datos_iniciales(self) -> None:
        cur = self._conn.cursor()
        if cur.execute("SELECT id FROM negocios WHERE id = ?", (self.negocio_id,)).fetchone():
            return

        cfg = _config_negocio_default(self.negocio_id)
        cur.execute(
            """
            INSERT INTO negocios (
                id, nombre_negocio, rubro, moneda, unidad_cantidad, etiqueta_cantidad,
                etiqueta_complejidad, telefono_dueno, margen_base, margen_minimo,
                descuento_maximo, mensaje_catalogo_intro
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cfg["id"], cfg["nombre_negocio"], cfg["rubro"], cfg["moneda"],
                cfg["unidad_cantidad"], cfg["etiqueta_cantidad"], cfg["etiqueta_complejidad"],
                cfg["telefono_dueno"], cfg["margen_base"], cfg["margen_minimo"],
                cfg["descuento_maximo"], cfg["mensaje_catalogo_intro"],
            ),
        )

        palabras_base = {
            "desarrollo_web": ["web", "pagina", "página", "sitio", "landing"],
            "automatizacion_ia": ["automatiz", "bot", "ia", "inteligencia artificial"],
            "ecommerce": ["tienda", "carrito", "ecommerce", "e-commerce"],
            "app_movil": ["movil", "móvil", "app", "android", "ios"],
            "consultoria_cloud": ["cloud", "nube", "devops", "servidor"],
        }
        for clave, info in CATALOGO_SERVICIOS.items():
            cur.execute(
                """
                INSERT INTO servicios (
                    negocio_id, clave, servicio_id_num, nombre, descripcion,
                    tarifa_unitaria, palabras_clave, requiere_complejidad,
                    costo_adicional_mantenimiento
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.negocio_id, clave, int(info["id"]), str(info["nombre"]),
                    str(info["descripcion"]), float(info["tarifa_hora_cop"]),
                    json.dumps(palabras_base.get(clave, [clave]), ensure_ascii=False),
                    1, 380_000.0,
                ),
            )

        for p in _promos_default():
            cur.execute(
                "INSERT INTO promociones_vigentes (negocio_id, codigo, descripcion, descuento_max, activa, condiciones) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.negocio_id,
                    p["codigo"],
                    p["descripcion"],
                    p["descuento_max"],
                    1 if p["activa"] else 0,
                    json.dumps(p.get("condiciones", {}), ensure_ascii=False),
                ),
            )

        for _, r in generar_tarifas_historicas(n_registros=45).iterrows():
            cur.execute(
                """
                INSERT INTO tarifas_historicas (
                    negocio_id, id_cotizacion, tipo_servicio, tipo_servicio_id,
                    complejidad, cantidad_unidades, urgencia_dias, incluye_adicional,
                    costo_base, costo_total, margen_estimado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.negocio_id, str(r["id_cotizacion"]), str(r["tipo_servicio"]),
                    int(r["tipo_servicio_id"]), int(r["complejidad"]), int(r["horas_estimadas"]),
                    int(r["urgencia_dias"]), int(r["incluye_mantenimiento"]),
                    float(r["costo_base_cop"]), float(r["costo_total_cop"]), float(r["margen_estimado"]),
                ),
            )

        for _, r in generar_historico_ofertas(n_registros=45).iterrows():
            cur.execute(
                """
                INSERT INTO historico_ofertas (
                    negocio_id, id_oferta, monto_cotizado, nivel_indecision,
                    complejidad, fidelidad_cliente, porcentaje_descuento_optimo,
                    cerrado_exitosamente
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.negocio_id, str(r["id_oferta"]), float(r["monto_cotizado"]),
                    float(r["nivel_indecision"]), int(r["complejidad"]),
                    int(r["fidelidad_cliente"]), float(r["porcentaje_descuento_optimo"]),
                    int(r["cerrado_exitosamente"]),
                ),
            )

        self._conn.commit()

    def obtener_configuracion_negocio(self) -> Dict[str, Any]:
        row = self._conn.execute("SELECT * FROM negocios WHERE id = ?", (self.negocio_id,)).fetchone()
        return dict(row) if row is not None else _config_negocio_default(self.negocio_id)

    def obtener_catalogo_servicios(self) -> Dict[str, Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM servicios WHERE negocio_id = ? ORDER BY servicio_id_num",
            (self.negocio_id,),
        ).fetchall()
        return _mapear_catalogo(rows)

    def obtener_promociones_vigentes(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM promociones_vigentes WHERE negocio_id = ?",
            (self.negocio_id,),
        ).fetchall()
        return _mapear_promociones(rows)

    def obtener_tarifas_historicas(self) -> pd.DataFrame:
        rows = self._conn.execute(
            "SELECT * FROM tarifas_historicas WHERE negocio_id = ?",
            (self.negocio_id,),
        ).fetchall()
        return _mapear_tarifas_df(rows)

    def obtener_historico_ofertas(self) -> pd.DataFrame:
        rows = self._conn.execute(
            "SELECT * FROM historico_ofertas WHERE negocio_id = ?",
            (self.negocio_id,),
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def listar_perfiles_clientes(self) -> Dict[str, Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM perfiles_clientes WHERE negocio_id = ?",
            (self.negocio_id,),
        ).fetchall()
        return _mapear_perfiles(rows)

    def guardar_perfil_cliente(self, perfil: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO perfiles_clientes (
                negocio_id, id_cliente, nombre, canal, historial_interacciones,
                tono_actual, fidelidad, ultima_cotizacion, ultima_complejidad,
                parametros_parciales, esperando_parametros_cotizacion,
                promocion_aplicada, automatizacion_activa
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(negocio_id, id_cliente) DO UPDATE SET
                nombre = excluded.nombre,
                canal = excluded.canal,
                historial_interacciones = excluded.historial_interacciones,
                tono_actual = excluded.tono_actual,
                fidelidad = excluded.fidelidad,
                ultima_cotizacion = excluded.ultima_cotizacion,
                ultima_complejidad = excluded.ultima_complejidad,
                parametros_parciales = excluded.parametros_parciales,
                esperando_parametros_cotizacion = excluded.esperando_parametros_cotizacion,
                promocion_aplicada = excluded.promocion_aplicada,
                automatizacion_activa = excluded.automatizacion_activa
            """,
            (
                self.negocio_id,
                str(perfil["id_cliente"]),
                str(perfil.get("nombre", "Cliente Potencial")),
                str(perfil.get("canal", "WhatsApp")),
                json.dumps(perfil.get("historial_interacciones", []), ensure_ascii=False),
                str(perfil.get("tono_actual", "neutral")),
                int(perfil.get("fidelidad", 1)),
                perfil.get("ultima_cotizacion"),
                int(perfil.get("ultima_complejidad", 3)),
                json.dumps(perfil.get("parametros_parciales", {}), ensure_ascii=False),
                1 if perfil.get("esperando_parametros_cotizacion") else 0,
                perfil.get("promocion_aplicada"),
                1 if perfil.get("automatizacion_activa", True) else 0,
            ),
        )
        self._conn.commit()

    def listar_alertas_humanas(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM alertas_humanas WHERE negocio_id = ? ORDER BY id",
            (self.negocio_id,),
        ).fetchall()
        return _mapear_alertas(rows)

    def registrar_alerta_humana(self, alerta: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO alertas_humanas (
                negocio_id, timestamp, id_cliente, nombre, canal,
                motivo_critico, historial_charla, estado, resuelta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.negocio_id,
                str(alerta.get("timestamp", "")),
                str(alerta.get("id_cliente", "")),
                str(alerta.get("nombre", "")),
                str(alerta.get("canal", "")),
                str(alerta.get("motivo_critico", "")),
                json.dumps(alerta.get("historial_charla", []), ensure_ascii=False),
                str(alerta.get("estado", "")),
                1 if alerta.get("resuelta") else 0,
            ),
        )
        self._conn.commit()

    def resolver_alertas_cliente(self, id_cliente: str) -> None:
        self._conn.execute(
            "UPDATE alertas_humanas SET resuelta = 1 WHERE negocio_id = ? AND id_cliente = ?",
            (self.negocio_id, id_cliente),
        )
        self._conn.commit()

    def listar_agendamientos_activos(
        self, fecha_inicio: str, fecha_fin: str
    ) -> List[Dict[str, Any]]:
        inicio_solicitado = _fecha_utc(fecha_inicio)
        fin_solicitado = _fecha_utc(fecha_fin)
        rows = self._conn.execute(
            """
            SELECT * FROM agendamientos
            WHERE negocio_id = ? AND estado IN ('pendiente', 'confirmada')
            """,
            (self.negocio_id,),
        ).fetchall()
        return [
            _mapear_agendamiento(dict(row))
            for row in rows
            if inicio_solicitado < _fecha_utc(row["fecha_fin"])
            and fin_solicitado > _fecha_utc(row["fecha_inicio"])
        ]

    def registrar_agendamiento(self, agendamiento: Dict[str, Any]) -> Dict[str, Any]:
        inicio = str(agendamiento["fecha_inicio"])
        fin = str(agendamiento["fecha_fin"])
        inicio_utc = _fecha_utc(inicio)
        fin_utc = _fecha_utc(fin)
        if fin_utc <= inicio_utc:
            raise ValueError("La fecha final debe ser posterior a la fecha inicial.")

        registro = {
            "id": str(agendamiento.get("id") or uuid.uuid4()),
            "negocio_id": self.negocio_id,
            "id_cliente": str(agendamiento["id_cliente"]),
            "nombre_cliente": str(agendamiento["nombre_cliente"]),
            "canal": str(agendamiento["canal"]),
            "servicio": agendamiento.get("servicio"),
            "fecha_inicio": inicio,
            "fecha_fin": fin,
            "estado": str(agendamiento.get("estado", "confirmada")),
            "registrado_en": str(agendamiento["registrado_en"]),
            "google_calendar_estado": str(agendamiento["google_calendar_estado"]),
            "google_event_id": agendamiento.get("google_event_id"),
            "google_event_link": agendamiento.get("google_event_link"),
            "google_calendar_error": agendamiento.get("google_calendar_error"),
            "observaciones": agendamiento.get("observaciones"),
            "metadata": json.dumps(agendamiento.get("metadata", {}), ensure_ascii=False),
        }

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            activos = self._conn.execute(
                """
                SELECT fecha_inicio, fecha_fin FROM agendamientos
                WHERE negocio_id = ? AND estado IN ('pendiente', 'confirmada')
                """,
                (self.negocio_id,),
            ).fetchall()
            if any(
                inicio_utc < _fecha_utc(row["fecha_fin"])
                and fin_utc > _fecha_utc(row["fecha_inicio"])
                for row in activos
            ):
                raise ConflictoAgendamientoError(
                    "El horario solicitado se cruza con otra cita activa."
                )
            self._conn.execute(
                """
                INSERT INTO agendamientos (
                    id, negocio_id, id_cliente, nombre_cliente, canal, servicio,
                    fecha_inicio, fecha_fin, estado, registrado_en,
                    google_calendar_estado, google_event_id, google_event_link,
                    google_calendar_error, observaciones, metadata
                ) VALUES (
                    :id, :negocio_id, :id_cliente, :nombre_cliente, :canal, :servicio,
                    :fecha_inicio, :fecha_fin, :estado, :registrado_en,
                    :google_calendar_estado, :google_event_id, :google_event_link,
                    :google_calendar_error, :observaciones, :metadata
                )
                """,
                registro,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        fila = self._conn.execute(
            "SELECT * FROM agendamientos WHERE id = ?", (registro["id"],)
        ).fetchone()
        if fila is None:
            raise RuntimeError("No se pudo verificar el agendamiento registrado.")
        return _mapear_agendamiento(dict(fila))

    def actualizar_agendamiento(
        self, id_agendamiento: str, cambios: Dict[str, Any]
    ) -> Dict[str, Any]:
        permitidos = {
            "estado",
            "registrado_en",
            "google_calendar_estado",
            "google_event_id",
            "google_event_link",
            "google_calendar_error",
            "observaciones",
            "metadata",
        }
        valores = {key: value for key, value in cambios.items() if key in permitidos}
        if not valores:
            raise ValueError("No hay campos válidos para actualizar el agendamiento.")
        if "metadata" in valores:
            valores["metadata"] = json.dumps(valores["metadata"], ensure_ascii=False)
        valores["updated_at"] = datetime.now(timezone.utc).isoformat()
        assignments = ", ".join(f"{key} = :{key}" for key in valores)
        valores.update({"id": id_agendamiento, "negocio_id": self.negocio_id})
        self._conn.execute(
            f"""
            UPDATE agendamientos SET {assignments}
            WHERE id = :id AND negocio_id = :negocio_id
            """,
            valores,
        )
        self._conn.commit()
        fila = self._conn.execute(
            "SELECT * FROM agendamientos WHERE id = ? AND negocio_id = ?",
            (id_agendamiento, self.negocio_id),
        ).fetchone()
        if fila is None:
            raise RuntimeError("No se encontró el agendamiento para actualizar.")
        return _mapear_agendamiento(dict(fila))

    def obtener_agendamiento(
        self, id_agendamiento: str
    ) -> Optional[Dict[str, Any]]:
        fila = self._conn.execute(
            "SELECT * FROM agendamientos WHERE id = ? AND negocio_id = ?",
            (id_agendamiento, self.negocio_id),
        ).fetchone()
        return _mapear_agendamiento(dict(fila)) if fila is not None else None


class RepositorioSupabase(RepositorioEntorno):
    """
    Repositorio en la nube sobre PostgreSQL de Supabase usando PostgREST (HTTPS 443).
    Evita bloqueos del puerto 5432 en redes restringidas y funciona multi-negocio.
    """

    modo_bd = "Supabase Cloud (PostgreSQL)"

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        negocio_id: str = "tech_freelance",
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        super().__init__(negocio_id=negocio_id)
        self.base_rest = supabase_url.strip().rstrip("/").removesuffix("/rest/v1") + "/rest/v1"
        key_limpia = supabase_key.strip()
        self.headers = {
            "apikey": key_limpia,
            "Authorization": f"Bearer {key_limpia}",
            "Content-Type": "application/json",
        }
        self._client = http_client or httpx.Client(trust_env=False, timeout=12.0)

    def _get(self, tabla: str, params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        query = {"negocio_id": f"eq.{self.negocio_id}", **(params or {})}
        resp = self._client.get(
            f"{self.base_rest}/{tabla}", headers=self.headers, params=query
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def _upsert(self, tabla: str, payload: Dict[str, Any], on_conflict: str) -> None:
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"}
        resp = self._client.post(
            f"{self.base_rest}/{tabla}",
            headers=headers,
            params={"on_conflict": on_conflict},
            json=payload,
        )
        resp.raise_for_status()

    def obtener_configuracion_negocio(self) -> Dict[str, Any]:
        resp = self._client.get(
            f"{self.base_rest}/negocios",
            headers=self.headers,
            params={"id": f"eq.{self.negocio_id}", "select": "*"},
        )
        resp.raise_for_status()
        filas = resp.json()
        return dict(filas[0]) if isinstance(filas, list) and filas else _config_negocio_default(self.negocio_id)

    def obtener_catalogo_servicios(self) -> Dict[str, Dict[str, Any]]:
        return _mapear_catalogo(self._get("servicios", {"order": "servicio_id_num.asc"}))

    def obtener_promociones_vigentes(self) -> List[Dict[str, Any]]:
        return _mapear_promociones(self._get("promociones_vigentes"))

    def obtener_tarifas_historicas(self) -> pd.DataFrame:
        return _mapear_tarifas_df(self._get("tarifas_historicas"))

    def obtener_historico_ofertas(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("historico_ofertas"))

    def listar_perfiles_clientes(self) -> Dict[str, Dict[str, Any]]:
        return _mapear_perfiles(self._get("perfiles_clientes"))

    def guardar_perfil_cliente(self, perfil: Dict[str, Any]) -> None:
        self._upsert(
            "perfiles_clientes",
            {
                "negocio_id": self.negocio_id,
                "id_cliente": str(perfil["id_cliente"]),
                "nombre": str(perfil.get("nombre", "Cliente Potencial")),
                "canal": str(perfil.get("canal", "WhatsApp")),
                "historial_interacciones": perfil.get("historial_interacciones", []),
                "tono_actual": str(perfil.get("tono_actual", "neutral")),
                "fidelidad": int(perfil.get("fidelidad", 1)),
                "ultima_cotizacion": perfil.get("ultima_cotizacion"),
                "ultima_complejidad": int(perfil.get("ultima_complejidad", 3)),
                "parametros_parciales": perfil.get("parametros_parciales", {}),
                "esperando_parametros_cotizacion": bool(
                    perfil.get("esperando_parametros_cotizacion", False)
                ),
                "promocion_aplicada": perfil.get("promocion_aplicada"),
                "automatizacion_activa": bool(perfil.get("automatizacion_activa", True)),
            },
            on_conflict="negocio_id,id_cliente",
        )

    def listar_alertas_humanas(self) -> List[Dict[str, Any]]:
        return _mapear_alertas(self._get("alertas_humanas", {"order": "id.asc"}))

    def registrar_alerta_humana(self, alerta: Dict[str, Any]) -> None:
        headers = {**self.headers, "Prefer": "return=minimal"}
        resp = self._client.post(
            f"{self.base_rest}/alertas_humanas",
            headers=headers,
            json={
                "negocio_id": self.negocio_id,
                "timestamp": str(alerta.get("timestamp", "")),
                "id_cliente": str(alerta.get("id_cliente", "")),
                "nombre": str(alerta.get("nombre", "")),
                "canal": str(alerta.get("canal", "")),
                "motivo_critico": str(alerta.get("motivo_critico", "")),
                "historial_charla": alerta.get("historial_charla", []),
                "estado": str(alerta.get("estado", "")),
                "resuelta": bool(alerta.get("resuelta", False)),
            },
        )
        resp.raise_for_status()

    def resolver_alertas_cliente(self, id_cliente: str) -> None:
        headers = {**self.headers, "Prefer": "return=minimal"}
        resp = self._client.patch(
            f"{self.base_rest}/alertas_humanas",
            headers=headers,
            params={
                "negocio_id": f"eq.{self.negocio_id}",
                "id_cliente": f"eq.{id_cliente}",
            },
            json={"resuelta": True},
        )
        resp.raise_for_status()

    def listar_agendamientos_activos(
        self, fecha_inicio: str, fecha_fin: str
    ) -> List[Dict[str, Any]]:
        inicio = _fecha_utc(fecha_inicio).isoformat()
        fin = _fecha_utc(fecha_fin).isoformat()
        return [
            _mapear_agendamiento(row)
            for row in self._get(
                "agendamientos",
                {
                    "select": "*",
                    "estado": "in.(pendiente,confirmada)",
                    "fecha_inicio": f"lt.{fin}",
                    "fecha_fin": f"gt.{inicio}",
                },
            )
        ]

    def registrar_agendamiento(self, agendamiento: Dict[str, Any]) -> Dict[str, Any]:
        inicio = str(agendamiento["fecha_inicio"])
        fin = str(agendamiento["fecha_fin"])
        inicio_utc = _fecha_utc(inicio)
        fin_utc = _fecha_utc(fin)
        if fin_utc <= inicio_utc:
            raise ValueError("La fecha final debe ser posterior a la fecha inicial.")
        if self.listar_agendamientos_activos(inicio, fin):
            raise ConflictoAgendamientoError(
                "El horario solicitado se cruza con otra cita activa."
            )

        valores_requeridos = {
            "id_cliente": agendamiento.get("id_cliente"),
            "nombre_cliente": agendamiento.get("nombre_cliente"),
            "canal": agendamiento.get("canal"),
        }
        if any(not str(valor or "").strip() for valor in valores_requeridos.values()):
            raise ValueError(
                "El agendamiento requiere id_cliente, nombre_cliente y canal."
            )
        estado = str(agendamiento.get("estado", "confirmada"))
        registrado_en = str(agendamiento.get("registrado_en", ""))
        google_estado = str(agendamiento.get("google_calendar_estado", ""))
        if estado not in {"pendiente", "confirmada", "cancelada", "error"}:
            raise ValueError("Estado de agendamiento no permitido.")
        if registrado_en not in {"supabase_y_google_calendar", "solo_supabase"}:
            raise ValueError("Valor de registrado_en no permitido.")
        if google_estado not in {
            "pendiente",
            "sincronizado",
            "no_configurado",
            "error",
        }:
            raise ValueError("Estado de Google Calendar no permitido.")

        payload = {
            "id": str(agendamiento.get("id") or uuid.uuid4()),
            "negocio_id": self.negocio_id,
            "id_cliente": str(agendamiento["id_cliente"]).strip(),
            "nombre_cliente": str(agendamiento["nombre_cliente"]).strip(),
            "canal": str(agendamiento["canal"]).strip(),
            "servicio": agendamiento.get("servicio"),
            "fecha_inicio": _fecha_bogota(inicio).isoformat(),
            "fecha_fin": _fecha_bogota(fin).isoformat(),
            "estado": estado,
            "registrado_en": registrado_en,
            "google_calendar_estado": google_estado,
            "google_event_id": agendamiento.get("google_event_id"),
            "google_event_link": agendamiento.get("google_event_link"),
            "google_calendar_error": agendamiento.get("google_calendar_error"),
            "observaciones": agendamiento.get("observaciones"),
            "metadata": agendamiento.get("metadata", {}),
        }
        headers = {**self.headers, "Prefer": "return=representation"}
        resp = self._client.post(
            f"{self.base_rest}/agendamientos", headers=headers, json=payload
        )
        if _es_conflicto_agendamiento_supabase(resp):
            raise ConflictoAgendamientoError(
                "El horario fue reservado por otra solicitud simultánea."
            ) from None
        resp.raise_for_status()
        filas = resp.json()
        if not isinstance(filas, list) or not filas:
            raise RuntimeError("Supabase no confirmó el INSERT del agendamiento.")
        return _mapear_agendamiento(filas[0])

    def actualizar_agendamiento(
        self, id_agendamiento: str, cambios: Dict[str, Any]
    ) -> Dict[str, Any]:
        permitidos = {
            "estado",
            "registrado_en",
            "google_calendar_estado",
            "google_event_id",
            "google_event_link",
            "google_calendar_error",
            "observaciones",
            "metadata",
        }
        payload = {key: value for key, value in cambios.items() if key in permitidos}
        if not payload:
            raise ValueError("No hay campos válidos para actualizar el agendamiento.")
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        resp = self._client.patch(
            f"{self.base_rest}/agendamientos",
            headers={**self.headers, "Prefer": "return=representation"},
            params={
                "negocio_id": f"eq.{self.negocio_id}",
                "id": f"eq.{id_agendamiento}",
            },
            json=payload,
        )
        resp.raise_for_status()
        filas = resp.json()
        if not isinstance(filas, list) or not filas:
            raise RuntimeError("Supabase no confirmó el UPDATE del agendamiento.")
        return _mapear_agendamiento(filas[0])

    def obtener_agendamiento(
        self, id_agendamiento: str
    ) -> Optional[Dict[str, Any]]:
        filas = self._get(
            "agendamientos",
            {"id": f"eq.{id_agendamiento}", "select": "*"},
        )
        return _mapear_agendamiento(filas[0]) if filas else None


def crear_repositorio(
    supabase_url: str = "",
    supabase_key: str = "",
    negocio_id: str = "tech_freelance",
    sqlite_path: str | Path = ":memory:",
) -> RepositorioEntorno:
    """
    Construye RepositorioSupabase si existen credenciales en .env; en caso
    contrario inicializa RepositorioSQLite como respaldo local o para pruebas.
    """
    if supabase_url.strip() and supabase_key.strip():
        return RepositorioSupabase(
            supabase_url=supabase_url.strip(),
            supabase_key=supabase_key.strip(),
            negocio_id=negocio_id,
        )
    return RepositorioSQLite(ruta_db=sqlite_path, negocio_id=negocio_id)
