"""
Capa de persistencia desacoplada para EntornoVirtualAtencion.
Soporta Supabase (PostgreSQL en la nube via REST HTTPS) y SQLite local de respaldo.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""

from src.persistencia.repositorio import (
    RepositorioEntorno,
    RepositorioSQLite,
    RepositorioSupabase,
    crear_repositorio,
)

__all__ = [
    "RepositorioEntorno",
    "RepositorioSQLite",
    "RepositorioSupabase",
    "crear_repositorio",
]
