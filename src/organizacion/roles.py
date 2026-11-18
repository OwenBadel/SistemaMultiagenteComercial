"""
Definicion formal de los 5 Roles MAS-ML (<<role>>) del SistemaMultiagenteComercial
con sus Deberes, Derechos y Restricciones exactos segun el Diagrama de Clases
y las tablas conceptuales del proyecto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RolMASML:
    """Clase base para los roles organizacionales MAS-ML (<<role>>)."""

    nombre: str
    deberes: List[str] = field(default_factory=list)
    derechos: List[str] = field(default_factory=list)
    restricciones: List[str] = field(default_factory=list)


@dataclass
class Coordinador(RolMASML):
    """<<role>> Coordinador â€” Orquestador de turnos, utilidades y control humano."""

    nombre: str = "Coordinador"
    deberes: List[str] = field(
        default_factory=lambda: [
            "Evaluar cotizaciones y decidir avance",
            "Asignar caso comercial con minima latencia",
            "Intermediar promociones hacia Ventas",
            "Suspender sistema y alertar al humano",
        ]
    )
    derechos: List[str] = field(
        default_factory=lambda: [
            "Delegar turnos exclusivos de atencion",
            "Solicitar calculo interno a especialistas",
            "Pausar automatizacion del chat",
        ]
    )
    restricciones: List[str] = field(
        default_factory=lambda: [
            "No formula respuestas persuasivas ni conversa directamente sobre ventas o precios con el cliente final",
        ]
    )


@dataclass
class EjecutivoVentas(RolMASML):
    """<<role>> EjecutivoVentas â€” Ãšnico interlocutor conversacional con el cliente."""

    nombre: str = "EjecutivoVentas"
    deberes: List[str] = field(
        default_factory=lambda: [
            "Mantener interaccion fluida y perfilar cliente",
            "Comunicar promociones aprobadas",
            "Escalar caso al Coordinador",
        ]
    )
    derechos: List[str] = field(
        default_factory=lambda: [
            "Atender inicialmente en el chat",
            "Enviar catalogos visuales",
            "Presentar descuentos preaprobados",
        ]
    )
    restricciones: List[str] = field(
        default_factory=lambda: [
            "No puede calcular presupuestos finales",
            "No puede diseÃ±ar o aprobar descuentos por cuenta propia",
            "No puede escribir citas en el calendario",
        ]
    )


@dataclass
class LiquidadorCotizaciones(RolMASML):
    """<<role>> LiquidadorCotizaciones â€” CÃ¡lculo econÃ³mico basado en tarifas y ML."""

    nombre: str = "LiquidadorCotizaciones"
    deberes: List[str] = field(
        default_factory=lambda: [
            "Garantizar exactitud en calculo del presupuesto",
            "Presentar desglose claro de costos",
        ]
    )
    derechos: List[str] = field(
        default_factory=lambda: [
            "Consultar BD de cotizaciones historicas",
            "Solicitar parametros y emitir cotizacion",
        ]
    )
    restricciones: List[str] = field(
        default_factory=lambda: [
            "No puede fijar presupuestos definitivos si faltan variables obligatorias",
            "No puede reservar fechas de atencion",
        ]
    )


@dataclass
class AnalistaPromociones(RolMASML):
    """<<role>> AnalistaPromociones â€” EvaluaciÃ³n de descuentos cuidando rentabilidad."""

    nombre: str = "AnalistaPromociones"
    deberes: List[str] = field(
        default_factory=lambda: [
            "Determinar y proponer incentivos",
            "Proteger margen de ganancia del profesional",
        ]
    )
    derechos: List[str] = field(
        default_factory=lambda: [
            "Consultar historico y tabla de margenes",
            "Proponer descuento al Coordinador",
        ]
    )
    restricciones: List[str] = field(
        default_factory=lambda: [
            "No puede interactuar ni enviar promociones directamente al chat del cliente",
            "No puede exceder los margenes maximos de descuento configurados",
        ]
    )


@dataclass
class GestorAgenda(RolMASML):
    """<<role>> GestorAgenda â€” AdministraciÃ³n y prevenciÃ³n de cruces en calendario."""

    nombre: str = "GestorAgenda"
    deberes: List[str] = field(
        default_factory=lambda: [
            "Prevenir superposiciones horarias",
            "Respetar jornadas y bloques de atencion",
        ]
    )
    derechos: List[str] = field(
        default_factory=lambda: [
            "Consultar disponibilidad en tiempo real",
            "Administrar y organizar el calendario",
        ]
    )
    restricciones: List[str] = field(
        default_factory=lambda: [
            "No puede modificar, consultar datos privados ni cancelar eventos preexistentes de otros clientes",
        ]
    )

