"""
Capa de persistencia desacoplada para EntornoVirtualAtencion.
Soporta Supabase (PostgreSQL en la nube via REST HTTPS) y SQLite local de respaldo.

Autores: Owen Badel, Maverick Diaz, Jesus Viloria
"""

from src.persistencia.repositorio import (
    CATALOGO_DEFAULT_SEED,
    RepositorioEntorno,
    RepositorioSQLite,
    RepositorioSupabase,
    crear_repositorio,
)

__all__ = [
    "CATALOGO_DEFAULT_SEED",
    "RepositorioEntorno",
    "RepositorioSQLite",
    "RepositorioSupabase",
    "crear_repositorio",
]
