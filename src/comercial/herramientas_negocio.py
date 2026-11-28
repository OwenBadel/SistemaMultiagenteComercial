"""
Herramientas deterministas (roles especialistas) invocadas por Google Gemini Flash
via Tool Calling, estrictamente aisladas por `negocio_id`.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.comercial.configuracion import ConfiguracionComercial
from src.comercial.repositorio_multitenant import (
    ConflictoAgendaError,
    RepositorioMultiTenant,
    normalizar_telefono,
    resolver_zona_horaria_segura,
)
from src.persistencia.google_calendar import ClienteGoogleCalendar, ErrorGoogleCalendar

_NOMBRES_DIAS = {
    1: "lunes",
    2: "martes",
    3: "miercoles",
    4: "jueves",
    5: "viernes",
    6: "sabado",
    7: "domingo",
}


def _parsear_hhmm(valor: str, default: Tuple[int, int]) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", str(valor or ""))
    if not m:
        return default
    h, mn = int(m.group(1)), int(m.group(2))
    if 0 <= h <= 23 and 0 <= mn <= 59:
        return h, mn
    return default


def _formatear_horario_legible(horario: Dict[str, Any]) -> str:
    dias = sorted(set(int(d) for d in horario.get("dias", [1, 2, 3, 4, 5, 6])))
    nombres = [_NOMBRES_DIAS.get(d, str(d)) for d in dias]
    inicio = str(horario.get("inicio", "08:00"))
    fin = str(horario.get("fin", "18:00"))
    pausas = horario.get("pausas") or []
    txt_pausas = ""
    if pausas:
        tramos = [f"{p[0]}-{p[1]}" for p in pausas if isinstance(p, (list, tuple)) and len(p) == 2]
        if tramos:
            txt_pausas = f" (descanso: {', '.join(tramos)})"
    return f"{', '.join(nombres)} de {inicio} a {fin}{txt_pausas}"


def construir_cliente_calendar_negocio(
    negocio: Dict[str, Any],
    config: ConfiguracionComercial,
    duracion_minutos: Optional[int] = None,
) -> Optional[ClienteGoogleCalendar]:
    """Construye un cliente de Google Calendar independiente para este negocio si tiene calendar_id."""
    cal_id = str(negocio.get("google_calendar_id") or "").strip()
    sa_json = str(config.google_service_account_json or "").strip()
    if not cal_id or not sa_json:
        return None
    horario = negocio.get("horario_laboral") or {}
    h_ini, _ = _parsear_hhmm(str(horario.get("inicio", "08:00")), (8, 0))
    h_fin, _ = _parsear_hhmm(str(horario.get("fin", "18:00")), (18, 0))
    duracion = int(duracion_minutos or horario.get("duracion_min") or 60)
    tz_nombre = str(negocio.get("zona_horaria") or config.google_timezone or "America/Bogota")
    resolver_zona_horaria_segura(tz_nombre)
    return ClienteGoogleCalendar(
        calendar_id=cal_id,
        service_account_json=sa_json,
        timezone=tz_nombre,
        duracion_minutos=max(15, duracion),
        hora_inicio_laboral=h_ini,
        hora_fin_laboral=h_fin,
    )


class ContextoHerramientasNegocio:
    """Ejecutor auditable de las 5 herramientas especialistas aisladas a un negocio y cliente."""

    def __init__(
        self,
        repositorio: RepositorioMultiTenant,
        negocio: Dict[str, Any],
        perfil: Dict[str, Any],
        canal: str,
        config: ConfiguracionComercial,
        fabrica_calendar: Optional[
            Callable[[Dict[str, Any], ConfiguracionComercial], Optional[ClienteGoogleCalendar]]
        ] = None,
    ) -> None:
        self.repo = repositorio
        self.negocio = negocio
        self.negocio_id = str(negocio["id"])
        self.perfil = perfil
        self.id_cliente = str(perfil["id_cliente"])
        self.canal = canal
        self.config = config
        self.tz = resolver_zona_horaria_segura(
            str(negocio.get("zona_horaria") or config.google_timezone or "America/Bogota")
        )
        self._fabrica_calendar = fabrica_calendar or construir_cliente_calendar_negocio
        self.herramientas_ejecutadas: List[Dict[str, Any]] = []
        self.escalamiento_disparado: Optional[Dict[str, Any]] = None
        self.cita_confirmada: Optional[Dict[str, Any]] = None

    def _registrar_traza(self, nombre: str, entrada: Dict[str, Any], salida: Dict[str, Any]) -> Dict[str, Any]:
        self.herramientas_ejecutadas.append(
            {"herramienta": nombre, "entrada": entrada, "salida": salida}
        )
        return salida

    # 1. Especialista Catalogo
    def consultar_servicios(self) -> Dict[str, Any]:
        servicios = self.repo.listar_servicios(self.negocio_id, solo_activos=True)
        promos = self.repo.listar_promociones(self.negocio_id, solo_activas=True)
        salida = {
            "negocio": self.negocio["nombre_negocio"],
            "rubro": self.negocio["rubro"],
            "moneda": self.negocio["moneda"],
            "servicios": [
                {
                    "clave": s["clave"],
                    "nombre": s["nombre"],
                    "descripcion": s["descripcion"],
                    "precio": float(s["tarifa_unitaria"]),
                    "duracion_minutos": int(s.get("duracion_minutos") or 60),
                }
                for s in servicios
            ],
            "promociones_activas": [
                {
                    "codigo": p["codigo"],
                    "descripcion": p["descripcion"],
                    "descuento_max_porciento": float(p["descuento_max"]),
                    "condiciones": p.get("condiciones") or {},
                }
                for p in promos
            ],
        }
        return self._registrar_traza("consultar_servicios", {}, salida)

    # 2. Especialista Cotizaciones & Promociones (aritmetica 100% exacta en Python)
    def calcular_cotizacion(
        self,
        items: List[Dict[str, Any]],
        aplicar_promocion: bool = False,
        diseno_complejo: bool = False,
        incluir_domicilio_base: bool = False,
        pago_con_tarjeta: bool = False,
    ) -> Dict[str, Any]:
        servicios = {
            s["clave"]: s for s in self.repo.listar_servicios(self.negocio_id, solo_activos=True)
        }
        servicios_por_nombre = {s["nombre"].strip().casefold(): s for s in servicios.values()}
        reglas_neg = (self.negocio.get("reglas_y_faq") or {}).get("reglas_cotizacion") or {}
        categorias_unicas = [str(c).casefold() for c in (reglas_neg.get("una_por_categoria") or [])]
        conteo_categorias: Dict[str, int] = {}

        detalle = []
        subtotal = 0.0
        total_unidades = 0
        duracion_total = 0
        claves_elegidas: List[str] = []

        for item in items or []:
            clave_raw = str(item.get("servicio_clave") or item.get("clave") or "").strip()
            cantidad = max(1, int(item.get("cantidad") or 1))
            srv = servicios.get(clave_raw) or servicios_por_nombre.get(clave_raw.casefold())
            if not srv:
                return self._registrar_traza(
                    "calcular_cotizacion",
                    {"items": items, "aplicar_promocion": aplicar_promocion},
                    {
                        "error": f"El servicio '{clave_raw}' no existe en el catalogo activo.",
                        "servicios_disponibles": list(servicios.keys()),
                    },
                )
            if categorias_unicas:
                palabras = [str(p).casefold() for p in (srv.get("palabras_clave") or [])]
                for cat in categorias_unicas:
                    if cat in palabras:
                        conteo_categorias[cat] = conteo_categorias.get(cat, 0) + cantidad
                        if conteo_categorias[cat] > 1:
                            return self._registrar_traza(
                                "calcular_cotizacion",
                                {"items": items, "aplicar_promocion": aplicar_promocion},
                                {
                                    "error": (
                                        f"Solo se permite combinar un servicio de '{cat}' por cita "
                                        "(por ejemplo, 1 de manos y 1 de pies), salvo autorizacion manual de la propietaria."
                                    )
                                },
                            )
            precio_u = float(srv["tarifa_unitaria"])
            linea = round(precio_u * cantidad, 2)
            subtotal += linea
            total_unidades += cantidad
            duracion_total += int(srv.get("duracion_minutos") or 60) * cantidad
            claves_elegidas.append(srv["clave"])
            detalle.append(
                {
                    "clave": srv["clave"],
                    "nombre": srv["nombre"],
                    "cantidad": cantidad,
                    "precio_unitario": precio_u,
                    "subtotal_linea": linea,
                    "duracion_minutos": int(srv.get("duracion_minutos") or 60) * cantidad,
                }
            )

        if not detalle:
            return self._registrar_traza(
                "calcular_cotizacion",
                {"items": items, "aplicar_promocion": aplicar_promocion},
                {"error": "Debes indicar al menos un servicio del catalogo para cotizar."},
            )

        adicionales: List[Dict[str, Any]] = []
        if diseno_complejo and float(reglas_neg.get("costo_diseno_complejo") or 0) > 0:
            val_dis = float(reglas_neg["costo_diseno_complejo"])
            subtotal += val_dis
            duracion_total = max(duracion_total, 120)
            adicionales.append({"concepto": "Diseno complejo", "valor": val_dis})
        if incluir_domicilio_base and float(reglas_neg.get("costo_domicilio_minimo") or 0) > 0:
            val_dom = float(reglas_neg["costo_domicilio_minimo"])
            subtotal += val_dom
            adicionales.append({"concepto": "Domicilio base (confirmar segun distancia)", "valor": val_dom})

        descuento_pct = 0.0
        promo_aplicada = None
        if aplicar_promocion:
            promos = self.repo.listar_promociones(self.negocio_id, solo_activas=True)
            tope_negocio = float(self.negocio.get("descuento_maximo", 15.0))
            margen_base = float(self.negocio.get("margen_base", 0.35))
            margen_min = float(self.negocio.get("margen_minimo", 0.22))
            max_por_margen = max(0.0, round((margen_base - margen_min) * 100.0, 2))
            for p in promos:
                cond = p.get("condiciones") or {}
                cant_min = int(cond.get("cantidad_minima") or 1)
                cant_max = int(cond.get("cantidad_maxima") or 9999)
                servs_permitidos = cond.get("servicios") or []
                cumple_cant = cant_min <= total_unidades <= cant_max
                cumple_serv = not servs_permitidos or all(c in servs_permitidos for c in claves_elegidas)
                if cumple_cant and cumple_serv:
                    candidato = min(float(p["descuento_max"]), tope_negocio, max_por_margen)
                    if candidato > descuento_pct:
                        descuento_pct = candidato
                        promo_aplicada = {
                            "codigo": p["codigo"],
                            "descripcion": p["descripcion"],
                            "descuento_porciento": descuento_pct,
                        }

        descuento_valor = round(subtotal * (descuento_pct / 100.0), 2)
        total = round(subtotal - descuento_valor, 2)

        recargo_tarjeta_valor = 0.0
        recargo_pct = float(reglas_neg.get("recargo_tarjeta_pct") or 0.0)
        if pago_con_tarjeta and recargo_pct > 0:
            recargo_tarjeta_valor = round(total * (recargo_pct / 100.0), 2)
            total = round(total + recargo_tarjeta_valor, 2)

        anticipo_pct = float(reglas_neg.get("anticipo_porcentaje") or 0.0)
        anticipo_valor = round(total * (anticipo_pct / 100.0), 2) if anticipo_pct > 0 else 0.0

        self.perfil["ultima_cotizacion"] = total
        salida: Dict[str, Any] = {
            "moneda": self.negocio["moneda"],
            "detalle": detalle,
            "adicionales": adicionales,
            "subtotal": subtotal,
            "descuento_porciento": descuento_pct,
            "descuento_valor": descuento_valor,
            "recargo_tarjeta_valor": recargo_tarjeta_valor,
            "total": total,
            "anticipo_porcentaje": anticipo_pct,
            "anticipo_valor": anticipo_valor,
            "saldo_restante": round(total - anticipo_valor, 2),
            "duracion_total_minutos": duracion_total,
            "promocion_aplicada": promo_aplicada,
        }
        return self._registrar_traza(
            "calcular_cotizacion",
            {
                "items": items,
                "aplicar_promocion": aplicar_promocion,
                "diseno_complejo": diseno_complejo,
                "incluir_domicilio_base": incluir_domicilio_base,
                "pago_con_tarjeta": pago_con_tarjeta,
            },
            salida,
        )

    def _bloques_pausa_dia(self, dia_obj: date) -> List[Tuple[datetime, datetime]]:
        horario = self.negocio.get("horario_laboral") or {}
        bloques: List[Tuple[datetime, datetime]] = []
        for tramo in horario.get("pausas") or []:
            if isinstance(tramo, (list, tuple)) and len(tramo) == 2:
                p_ini_h, p_ini_m = _parsear_hhmm(str(tramo[0]), (12, 0))
                p_fin_h, p_fin_m = _parsear_hhmm(str(tramo[1]), (14, 0))
                dt_p_ini = datetime.combine(dia_obj, time(p_ini_h, p_ini_m), tzinfo=self.tz)
                dt_p_fin = datetime.combine(dia_obj, time(p_fin_h, p_fin_m), tzinfo=self.tz)
                if dt_p_fin > dt_p_ini:
                    bloques.append((dt_p_ini, dt_p_fin))
        return bloques

    def _obtener_ocupados_dia(
        self, dia_ini: datetime, dia_fin: datetime
    ) -> Tuple[List[Tuple[datetime, datetime]], Optional[str]]:
        horario = self.negocio.get("horario_laboral") or {}
        buffer_min = max(0, int(horario.get("buffer_minutos") or 0))
        ocupados: List[Tuple[datetime, datetime]] = list(self._bloques_pausa_dia(dia_ini.date()))
        for c in self.repo.listar_agendamientos(
            self.negocio_id,
            fecha_inicio=dia_ini.isoformat(),
            fecha_fin=dia_fin.isoformat(),
            solo_activos=True,
        ):
            try:
                c_ini = datetime.fromisoformat(str(c["fecha_inicio"]).replace("Z", "+00:00")).astimezone(self.tz)
                c_fin = datetime.fromisoformat(str(c["fecha_fin"]).replace("Z", "+00:00")).astimezone(self.tz)
                ocupados.append((c_ini, c_fin + timedelta(minutes=buffer_min)))
            except ValueError:
                continue

        cal_cliente = self._fabrica_calendar(self.negocio, self.config)
        error_cal: Optional[str] = None
        if cal_cliente is not None:
            try:
                for b_ini, b_fin in cal_cliente._consultar_freebusy(dia_ini, dia_fin):
                    ocupados.append((b_ini, b_fin + timedelta(minutes=buffer_min)))
            except ErrorGoogleCalendar as exc:
                error_cal = str(exc)
        return ocupados, error_cal

    # 3. Especialista Agenda - Consulta de horarios respetando jornada del negocio
    def consultar_horarios_disponibles(
        self, fecha: str, duracion_minutos: int = 0
    ) -> Dict[str, Any]:
        try:
            dia_obj = date.fromisoformat(str(fecha).strip()[:10])
        except ValueError:
            return self._registrar_traza(
                "consultar_horarios_disponibles",
                {"fecha": fecha, "duracion_minutos": duracion_minutos},
                {"error": "Formato de fecha invalido. Usa YYYY-MM-DD."},
            )

        ahora = datetime.now(self.tz)
        if dia_obj < ahora.date():
            return self._registrar_traza(
                "consultar_horarios_disponibles",
                {"fecha": fecha, "duracion_minutos": duracion_minutos},
                {"fecha": dia_obj.isoformat(), "disponible": False, "error": "La fecha indicada ya paso."},
            )

        horario = self.negocio.get("horario_laboral") or {}
        max_dias = horario.get("max_dias_anticipacion")
        if max_dias is not None and dia_obj > ahora.date() + timedelta(days=int(max_dias)):
            return self._registrar_traza(
                "consultar_horarios_disponibles",
                {"fecha": dia_obj.isoformat(), "duracion_minutos": duracion_minutos},
                {
                    "fecha": dia_obj.isoformat(),
                    "disponible": False,
                    "bloques_disponibles": [],
                    "error": f"Solo se permite reservar con maximo {int(max_dias)} dia(s) de anticipacion.",
                },
            )
        dias_lab = [int(d) for d in horario.get("dias", [1, 2, 3, 4, 5, 6])]
        if dia_obj.isoweekday() not in dias_lab:
            return self._registrar_traza(
                "consultar_horarios_disponibles",
                {"fecha": dia_obj.isoformat(), "duracion_minutos": duracion_minutos},
                {
                    "fecha": dia_obj.isoformat(),
                    "dia_laboral": False,
                    "horario_atencion": _formatear_horario_legible(horario),
                    "bloques_disponibles": [],
                    "mensaje": f"No atendemos los {_NOMBRES_DIAS.get(dia_obj.isoweekday(), '')}. Nuestro horario es {_formatear_horario_legible(horario)}.",
                },
            )

        h_ini, m_ini = _parsear_hhmm(str(horario.get("inicio", "08:00")), (8, 0))
        h_fin, m_fin = _parsear_hhmm(str(horario.get("fin", "18:00")), (18, 0))
        duracion = max(15, int(duracion_minutos or horario.get("duracion_min") or 60))

        jornada_ini = datetime.combine(dia_obj, time(h_ini, m_ini), tzinfo=self.tz)
        jornada_fin = datetime.combine(dia_obj, time(h_fin, m_fin), tzinfo=self.tz)
        if jornada_fin <= jornada_ini:
            return self._registrar_traza(
                "consultar_horarios_disponibles",
                {"fecha": dia_obj.isoformat(), "duracion_minutos": duracion},
                {
                    "fecha": dia_obj.isoformat(),
                    "dia_laboral": False,
                    "bloques_disponibles": [],
                    "error": "El horario laboral configurado no es valido.",
                },
            )

        ocupados, error_cal = self._obtener_ocupados_dia(jornada_ini, jornada_fin)

        bloques: List[str] = []
        paso_libre = timedelta(minutes=min(duracion, 30))
        delta_cita = timedelta(minutes=duracion)
        cursor = jornada_ini
        while cursor + delta_cita <= jornada_fin:
            fin_slot = cursor + delta_cita
            choques = [b_fin for b_ini, b_fin in ocupados if cursor < b_fin and fin_slot > b_ini]
            if choques:
                # Saltar exactamente al fin del bloque ocupado para no perder huecos inmediatos (ej. 10:00 tras 09:00-10:00)
                siguiente = max(choques)
                cursor = siguiente if siguiente > cursor else cursor + timedelta(minutes=15)
                continue
            if cursor > ahora + timedelta(minutes=15):
                bloques.append(cursor.strftime("%Y-%m-%d %H:%M"))
            cursor += paso_libre

        salida: Dict[str, Any] = {
            "fecha": dia_obj.isoformat(),
            "dia_laboral": True,
            "horario_atencion": _formatear_horario_legible(horario),
            "duracion_minutos": duracion,
            "bloques_disponibles": bloques[:16],
        }
        if error_cal:
            salida["advertencia_google_calendar"] = error_cal
        return self._registrar_traza(
            "consultar_horarios_disponibles",
            {"fecha": dia_obj.isoformat(), "duracion_minutos": duracion},
            salida,
        )

    # 4. Especialista Agenda - Registro definitivo de cita (Supabase + Google Calendar)
    def agendar_cita(
        self,
        fecha_hora: str,
        servicio_clave: str,
        nombre_cliente: str,
        telefono_cliente: str,
        cedula_cliente: str = "",
        observaciones: str = "",
    ) -> Dict[str, Any]:
        entrada = {
            "fecha_hora": fecha_hora,
            "servicio_clave": servicio_clave,
            "nombre_cliente": nombre_cliente,
            "telefono_cliente": telefono_cliente,
            "cedula_cliente": cedula_cliente,
        }
        nombre_limpio = " ".join((nombre_cliente or "").split())
        tel_digitos = normalizar_telefono(telefono_cliente)
        if len(nombre_limpio.split()) < 2:
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {"confirmada": False, "error": "Se requiere nombre y apellido del cliente para agendar."},
            )
        if not 7 <= len(tel_digitos) <= 15:
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {"confirmada": False, "error": "Se requiere un telefono de contacto valido (7 a 15 digitos)."},
            )

        servicios = {
            s["clave"]: s for s in self.repo.listar_servicios(self.negocio_id, solo_activos=True)
        }
        servicios_por_nombre = {s["nombre"].strip().casefold(): s for s in servicios.values()}
        srv = servicios.get(servicio_clave) or servicios_por_nombre.get(servicio_clave.strip().casefold())
        if not srv:
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {
                    "confirmada": False,
                    "error": f"Servicio '{servicio_clave}' no encontrado en el catalogo del negocio.",
                },
            )

        try:
            dt_ini = datetime.fromisoformat(str(fecha_hora).strip().replace("Z", "+00:00"))
            dt_ini = dt_ini.replace(tzinfo=self.tz) if dt_ini.tzinfo is None else dt_ini.astimezone(self.tz)
        except ValueError:
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {"confirmada": False, "error": "Formato de fecha_hora invalido. Usa 'YYYY-MM-DD HH:MM'."},
            )

        ahora = datetime.now(self.tz)
        if dt_ini <= ahora:
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {"confirmada": False, "error": "No se pueden agendar citas en fechas u horas pasadas."},
            )

        duracion = max(15, int(srv.get("duracion_minutos") or 60))
        dt_fin = dt_ini + timedelta(minutes=duracion)

        horario = self.negocio.get("horario_laboral") or {}
        max_dias = horario.get("max_dias_anticipacion")
        if max_dias is not None and dt_ini.date() > ahora.date() + timedelta(days=int(max_dias)):
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {
                    "confirmada": False,
                    "error": f"Solo se puede reservar con maximo {int(max_dias)} dia(s) de anticipacion.",
                },
            )

        dias_lab = [int(d) for d in horario.get("dias", [1, 2, 3, 4, 5, 6])]
        h_ini, m_ini = _parsear_hhmm(str(horario.get("inicio", "08:00")), (8, 0))
        h_fin, m_fin = _parsear_hhmm(str(horario.get("fin", "18:00")), (18, 0))
        jornada_ini = datetime.combine(dt_ini.date(), time(h_ini, m_ini), tzinfo=self.tz)
        jornada_fin = datetime.combine(dt_ini.date(), time(h_fin, m_fin), tzinfo=self.tz)

        if dt_ini.isoweekday() not in dias_lab or dt_ini < jornada_ini or dt_fin > jornada_fin:
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {
                    "confirmada": False,
                    "error": f"El horario solicitado esta fuera de nuestra jornada ({_formatear_horario_legible(horario)}).",
                },
            )

        for p_ini, p_fin in self._bloques_pausa_dia(dt_ini.date()):
            if dt_ini < p_fin and dt_fin > p_ini:
                alt = self.consultar_horarios_disponibles(dt_ini.date().isoformat(), duracion)
                return self._registrar_traza(
                    "agendar_cita",
                    entrada,
                    {
                        "confirmada": False,
                        "error": f"Ese horario cruza el descanso de jornada ({p_ini.strftime('%H:%M')} a {p_fin.strftime('%H:%M')}).",
                        "bloques_alternativos": alt.get("bloques_disponibles", [])[:4],
                    },
                )

        buffer_min = max(0, int(horario.get("buffer_minutos") or 0))
        if buffer_min > 0:
            ocupados_bd = self.repo.listar_agendamientos(
                self.negocio_id,
                fecha_inicio=(dt_ini - timedelta(minutes=buffer_min)).isoformat(),
                fecha_fin=(dt_fin + timedelta(minutes=buffer_min)).isoformat(),
                solo_activos=True,
            )
            for c in ocupados_bd:
                try:
                    c_ini = datetime.fromisoformat(str(c["fecha_inicio"]).replace("Z", "+00:00")).astimezone(self.tz)
                    c_fin = datetime.fromisoformat(str(c["fecha_fin"]).replace("Z", "+00:00")).astimezone(self.tz)
                    if dt_ini < c_fin + timedelta(minutes=buffer_min) and dt_fin + timedelta(minutes=buffer_min) > c_ini:
                        alt = self.consultar_horarios_disponibles(dt_ini.date().isoformat(), duracion)
                        return self._registrar_traza(
                            "agendar_cita",
                            entrada,
                            {
                                "confirmada": False,
                                "error": f"Se requiere un intervalo minimo de {buffer_min} minutos entre citas.",
                                "bloques_alternativos": alt.get("bloques_disponibles", [])[:4],
                            },
                        )
                except ValueError:
                    continue

        cal_cliente = self._fabrica_calendar(self.negocio, self.config)
        if cal_cliente is not None:
            cal_cliente.duracion_minutos = duracion

        gcal_estado = "no_configurado"
        gcal_event_id = None
        gcal_event_link = None
        registrado_en = "solo_supabase"

        if cal_cliente is not None:
            try:
                ocupados_gcal = cal_cliente._consultar_freebusy(dt_ini, dt_fin)
                if ocupados_gcal:
                    alt = self.consultar_horarios_disponibles(dt_ini.date().isoformat(), duracion)
                    return self._registrar_traza(
                        "agendar_cita",
                        entrada,
                        {
                            "confirmada": False,
                            "error": "Ese bloque horario ya esta ocupado en el calendario.",
                            "bloques_alternativos": alt.get("bloques_disponibles", [])[:4],
                        },
                    )
            except ErrorGoogleCalendar as exc:
                gcal_estado = "error"
                observaciones = f"{observaciones} | Google Calendar no verificado: {exc}".strip(" |")

        reglas_cot = (self.negocio.get("reglas_y_faq") or {}).get("reglas_cotizacion") or {}
        requiere_manual = bool(reglas_cot.get("requiere_confirmacion_manual_pago", False))
        estado_inicial = "pendiente" if requiere_manual else "confirmada"

        try:
            registro = self.repo.registrar_agendamiento(
                self.negocio_id,
                {
                    "id_cliente": self.id_cliente,
                    "nombre_cliente": nombre_limpio,
                    "canal": self.canal,
                    "servicio": srv["nombre"],
                    "fecha_inicio": dt_ini.isoformat(),
                    "fecha_fin": dt_fin.isoformat(),
                    "estado": estado_inicial,
                    "registrado_en": registrado_en,
                    "google_calendar_estado": gcal_estado,
                    "observaciones": observaciones,
                    "metadata": {
                        "servicio_clave": srv["clave"],
                        "telefono_cliente": tel_digitos,
                        "cedula_cliente": re.sub(r"\D+", "", str(cedula_cliente or "")),
                        "precio_referencia": float(srv["tarifa_unitaria"]),
                        "duracion_minutos": duracion,
                    },
                },
            )
        except ConflictoAgendaError:
            alt = self.consultar_horarios_disponibles(dt_ini.date().isoformat(), duracion)
            return self._registrar_traza(
                "agendar_cita",
                entrada,
                {
                    "confirmada": False,
                    "error": "Ese horario ya se encuentra reservado.",
                    "bloques_alternativos": alt.get("bloques_disponibles", [])[:4],
                },
            )

        if cal_cliente is not None and gcal_estado != "error":
            try:
                creado = cal_cliente.crear_evento(
                    fecha_hora=dt_ini.strftime("%Y-%m-%d %H:%M"),
                    cliente=nombre_limpio,
                    titulo=f"{srv['nombre']} - {nombre_limpio}",
                    descripcion=(
                        f"Negocio: {self.negocio['nombre_negocio']}\n"
                        f"Servicio: {srv['nombre']}\n"
                        f"Duracion: {duracion} minutos\n"
                        f"Cliente: {nombre_limpio}\n"
                        f"Telefono: {tel_digitos}\n"
                        f"Canal: {self.canal}"
                    ),
                )
                gcal_estado = "sincronizado"
                gcal_event_id = creado.get("evento_google_id") or creado.get("google_event_id")
                gcal_event_link = creado.get("html_link") or creado.get("google_event_link")
                registrado_en = "supabase_y_google_calendar"
            except ErrorGoogleCalendar:
                gcal_estado = "error"

            self.repo.actualizar_agendamiento(
                self.negocio_id,
                str(registro["id"]),
                {
                    "registrado_en": registrado_en,
                    "google_calendar_estado": gcal_estado,
                    "google_event_id": gcal_event_id,
                    "google_event_link": gcal_event_link,
                },
            )

        self.perfil["nombre"] = nombre_limpio
        anticipo_pct = float(reglas_cot.get("anticipo_porcentaje") or 0.0)
        self.cita_confirmada = {
            "confirmada": not requiere_manual,
            "estado_cita": estado_inicial,
            "requiere_confirmacion_manual_pago": requiere_manual,
            "anticipo_porcentaje": anticipo_pct,
            "agendamiento_id": registro["id"],
            "servicio": srv["nombre"],
            "fecha_hora": dt_ini.strftime("%Y-%m-%d %H:%M"),
            "fecha_fin": dt_fin.strftime("%Y-%m-%d %H:%M"),
            "duracion_minutos": duracion,
            "nombre_cliente": nombre_limpio,
            "telefono_cliente": tel_digitos,
            "google_calendar_estado": gcal_estado,
            "google_event_id": gcal_event_id,
            "google_event_link": gcal_event_link,
            "registrado_en": registrado_en,
        }
        return self._registrar_traza("agendar_cita", entrada, self.cita_confirmada)

    # 5. Especialista Control Human-in-the-Loop (pausa solo este chat y prepara alerta)
    def escalar_a_humano(
        self,
        motivo: str,
        resumen_para_dueno: str,
        tipo: str = "informacion_no_disponible",
    ) -> Dict[str, Any]:
        # Idempotencia por turno: evita alertas duplicadas si el modelo llama 2 veces
        if self.escalamiento_disparado is not None:
            return self._registrar_traza(
                "escalar_a_humano",
                {"motivo": motivo, "resumen_para_dueno": resumen_para_dueno, "tipo": tipo},
                {
                    "escalado": True,
                    "instruccion_respuesta": (
                        "Este chat ya fue escalado y pausado en este mismo turno. "
                        "Responde al cliente con amabilidad confirmando que un asesor humano le atendera pronto."
                    ),
                },
            )

        self.perfil["automatizacion_activa"] = False
        self.perfil["tono_actual"] = "hostil" if tipo == "reclamo_critico" else "interesado"
        self.repo.guardar_perfil(self.negocio_id, self.perfil)

        alerta = self.repo.registrar_alerta(
            self.negocio_id,
            {
                "id_cliente": self.id_cliente,
                "nombre": self.perfil.get("nombre", "Cliente"),
                "canal": self.canal,
                "motivo_critico": f"{tipo}: {motivo}",
                "resumen_ia": resumen_para_dueno,
                "historial_charla": self.perfil.get("historial_interacciones", [])[-10:],
                "estado": "pausado",
            },
        )

        tel_cliente = normalizar_telefono(self.id_cliente)
        enlace_wa = f"https://wa.me/{tel_cliente}" if len(tel_cliente) >= 10 else ""
        base_url = self.config.public_base_url or f"http://localhost:{self.config.port}"
        enlace_panel = f"{base_url}/?negocio={self.negocio_id}#alertas"

        mensaje_dueno = (
            f"ALERTA [{self.negocio['nombre_negocio']}] - Chat pausado para atencion humana\n"
            f"Cliente: {self.perfil.get('nombre', 'Cliente')} ({self.id_cliente})\n"
            f"Canal: {self.canal}\n"
            f"Resumen: {resumen_para_dueno}\n"
            f"Motivo: {motivo}\n"
            + (f"Responder en WhatsApp: {enlace_wa}\n" if enlace_wa else "")
            + f"Panel y reactivar bot: {enlace_panel}"
        )

        self.escalamiento_disparado = {
            "escalado": True,
            "pausado_humano": True,
            "tipo": tipo,
            "motivo": motivo,
            "resumen_para_dueno": resumen_para_dueno,
            "alerta_id": alerta.get("id"),
            "telefono_dueno": normalizar_telefono(self.negocio.get("telefono_dueno", "")),
            "mensaje_notificacion_dueno": mensaje_dueno,
            "enlace_whatsapp_cliente": enlace_wa,
            "enlace_panel_dueno": enlace_panel,
        }
        return self._registrar_traza(
            "escalar_a_humano",
            {"motivo": motivo, "resumen_para_dueno": resumen_para_dueno, "tipo": tipo},
            {
                "escalado": True,
                "instruccion_respuesta": (
                    "El chat ya fue pausado y el asesor humano recibira la notificacion. "
                    "Responde al cliente con amabilidad explicando que escalaste su caso "
                    "para que un asesor humano le responda por este mismo chat, sin inventar informacion."
                ),
            },
        )
