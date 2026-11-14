"""
Implementacion de la organizacion <<organization>> SistemaMultiagenteComercial
bajo el metamodelo MAS-ML: composicion de 5 roles (<<owns>>), relacion de habitat
(<<inhabit>>), control jerarquico (<<control>>), las 5 dependencias formales
(<<dependency>>) y reglas organizacionales de escalamiento y canal centralizado.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.entorno.entorno_virtual import EntornoVirtualAtencion
from src.organizacion.roles import (
    AnalistaPromociones,
    Coordinador,
    EjecutivoVentas,
    GestorAgenda,
    LiquidadorCotizaciones,
    RolMASML,
)


@dataclass(frozen=True)
class DependenciaFormal:
    """Representa una relación `[Rol que depende] <--- [Rol proveedor]`."""

    rol_dependiente: str
    rol_proveedor: str
    descripcion: str


class SistemaMultiagenteComercial:
    """
    <<organization>> SistemaMultiagenteComercial:
    - Posee por composición (<<owns>>) los 5 roles del sistema.
    - Habita (<<inhabit>>) el EntornoVirtualAtencion.
    - Aplica reglas de escalamiento obligatorio al Coordinador y canal
      centralizado de promociones.
    """

    REGLAS_ORGANIZACIONALES: Tuple[str, ...] = (
        "Escalamiento obligatorio al Coordinador ante cambio de intención o cotización",
        "Canal centralizado de promociones: AnalistaPromociones -> Coordinador -> EjecutivoVentas",
        "Interlocución única frente al cliente a cargo exclusivamente de EjecutivoVentas",
    )

    def __init__(self, entorno: EntornoVirtualAtencion) -> None:
        self.entorno: EntornoVirtualAtencion = entorno
        self.entorno.registrar_habitante("SistemaMultiagenteComercial")

        # Composición (<<owns>>) de los 5 roles exactos del diagrama MAS-ML
        self.rol_coordinador = Coordinador()
        self.rol_ventas = EjecutivoVentas()
        self.rol_cotizaciones = LiquidadorCotizaciones()
        self.rol_promociones = AnalistaPromociones()
        self.rol_agenda = GestorAgenda()

        self.roles: Dict[str, RolMASML] = {
            "Coordinador": self.rol_coordinador,
            "EjecutivoVentas": self.rol_ventas,
            "LiquidadorCotizaciones": self.rol_cotizaciones,
            "AnalistaPromociones": self.rol_promociones,
            "GestorAgenda": self.rol_agenda,
        }

        # Relaciones de <<control>> desde Coordinador hacia los 4 especialistas
        self.relaciones_control: Dict[str, List[str]] = {
            "Coordinador": [
                "EjecutivoVentas",
                "AnalistaPromociones",
                "LiquidadorCotizaciones",
                "GestorAgenda",
            ]
        }

        # Las 5 dependencias formales ([Rol que depende] <--- [Rol proveedor])
        self.dependencias_formales: List[DependenciaFormal] = [
            DependenciaFormal(
                rol_dependiente="Coordinador",
                rol_proveedor="LiquidadorCotizaciones",
                descripcion="El Coordinador depende del cálculo de Cotizaciones para evaluar y decidir el avance.",
            ),
            DependenciaFormal(
                rol_dependiente="Coordinador",
                rol_proveedor="AnalistaPromociones",
                descripcion="El Coordinador depende de la propuesta de descuento de Promociones para poder intermediarla.",
            ),
            DependenciaFormal(
                rol_dependiente="EjecutivoVentas",
                rol_proveedor="Coordinador",
                descripcion="Ventas depende del Coordinador para recibir la promoción aprobada y presentársela al cliente.",
            ),
            DependenciaFormal(
                rol_dependiente="GestorAgenda",
                rol_proveedor="LiquidadorCotizaciones",
                descripcion="Agenda depende de que exista una cotización aprobada antes de reservar en el calendario.",
            ),
            DependenciaFormal(
                rol_dependiente="Coordinador",
                rol_proveedor="GestorAgenda",
                descripcion="El Coordinador depende de la confirmación de la cita en el calendario.",
            ),
        ]

        # Registro de agentes jugando roles (<<play>>)
        self.agentes_por_rol: Dict[str, str] = {}

    def asignar_agente_a_rol(self, nombre_agente: str, nombre_rol: str) -> RolMASML:
        """Registra la relación <<play>> entre un agente y uno de los 5 roles."""
        if nombre_rol not in self.roles:
            raise ValueError(f"Rol '{nombre_rol}' no pertenece a la organización.")
        self.agentes_por_rol[nombre_rol] = nombre_agente
        self.entorno.registrar_habitante(nombre_agente)
        return self.roles[nombre_rol]

    def verificar_control(self, rol_supervisor: str, rol_supervisado: str) -> bool:
        """Valida si existe relación <<control>> entre dos roles."""
        return rol_supervisado in self.relaciones_control.get(rol_supervisor, [])

    def verificar_dependencia(self, rol_dependiente: str, rol_proveedor: str) -> bool:
        """Valida si existe una dependencia formal declarada `[dependiente] <--- [proveedor]`."""
        return any(
            d.rol_dependiente == rol_dependiente and d.rol_proveedor == rol_proveedor
            for d in self.dependencias_formales
        )

    def aplicar_regla_escalamiento(self, rol_origen: str, rol_destino: str) -> bool:
        """
        Regla 1: Ningún especialista responde directamente al cliente; EjecutivoVentas
        escala al Coordinador y este delega hacia Cotizaciones/Promociones/Agenda.
        """
        if rol_origen == "EjecutivoVentas" and rol_destino in {
            "LiquidadorCotizaciones",
            "AnalistaPromociones",
            "GestorAgenda",
        }:
            return False
        return True

    def aplicar_regla_canal_centralizado_promociones(
        self, rol_emisor: str, rol_receptor: str
    ) -> bool:
        """
        Regla 2: AnalistaPromociones nunca entrega ofertas directo a EjecutivoVentas
        ni al cliente; toda propuesta pasa por el Coordinador.
        """
        if rol_emisor == "AnalistaPromociones" and rol_receptor != "Coordinador":
            return False
        return True

    def validar_prerequisito_agendamiento(self, cotizacion_aprobada: bool) -> bool:
        """
        Hace cumplir la dependencia `GestorAgenda <--- LiquidadorCotizaciones`:
        no se puede reservar cita sin cotización aprobada previa.
        """
        return bool(cotizacion_aprobada)

