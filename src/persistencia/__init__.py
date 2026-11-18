"""
Capa de persistencia desacoplada para EntornoVirtualAtencion.
Soporta Supabase (PostgreSQL en la nube via REST HTTPS), Google Calendar API v3
y SQLite local de respaldo.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""

from src.persistencia.google_calendar import ClienteGoogleCalendar
from src.persistencia.repositorio import (
    RepositorioEntorno,
    RepositorioSQLite,
    RepositorioSupabase,
    crear_repositorio,
)

__all__ = [
    "ClienteGoogleCalendar",
    "RepositorioEntorno",
    "RepositorioSQLite",
    "RepositorioSupabase",
    "crear_repositorio",
]
