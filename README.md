# SistemaMultiagenteComercial

**Autores:** Owen Badel, Maverick Diaz, Jesus Viloria  
**Docente:** Juan Puello  
**Institucion:** Fundacion Internacional Colombo (Unicolombo)  
**Ecosistema tecnologico:** Python 3.12, SPADE (`spade`), Scikit-Learn (`scikit-learn`), FastAPI, Uvicorn, Botpress y Supabase (PostgreSQL en la nube).

---

## 1. Descripcion General

`SistemaMultiagenteComercial` implementa una arquitectura multiagente general basada en el modelo organizacional **MAS-ML** y el estandar **FIPA-ACL** sobre SPADE (con servidor XMPP embebido PyJabber), disenada para adaptarse a **cualquier trabajador independiente o negocio** leyendo su configuracion, catalogo de servicios, historicos ML, agenda y clientes directamente desde una base de datos en la nube (**Supabase PostgreSQL**, con respaldo automatico **SQLite** para pruebas locales).

Botpress actua unicamente como pasarela de canales (WhatsApp, Webchat, Messenger e Instagram) enviando cada mensaje entrante a `POST /api/botpress/mensaje`.

---

## 2. Arquitectura Organizacional y Entorno (MAS-ML)

- **Organizacion (`<<organization>> SistemaMultiagenteComercial`):** Define por composicion (`<<owns>>`) los 5 roles del sistema y aplica las reglas de escalamiento obligatorio hacia `Coordinador` y el canal centralizado de promociones.
- **Entorno (`<<environment>> EntornoVirtualAtencion`):** Espacio compartido conectado a `RepositorioEntorno` (`RepositorioSupabase` o `RepositorioSQLite`):
  - `config_negocio`: nombre del negocio, rubro, moneda, unidad de medida (`horas`, `sesiones`, etc.), `telefono_dueno`, margenes y topes de descuento leidos de BD.
  - `plataformas`: catalogo dinamico de servicios (`servicios` en BD) y registro multicanal.
  - `perfilesClientes`: memoria conversacional, tono, cotizacion vigente, parametros parciales y bandera `automatizacion_activa` por chat persistida en BD.
  - `tarifasYMargenes`: historico de tarifas y ofertas cargado desde BD para entrenar `RandomForestRegressor` y `KNeighborsRegressor`.
  - `agendaVirtual`: disponibilidad de bloques horarios persistida en BD.

### Agentes, Roles, Arquitecturas y Modelos ML

| Agente (`<<agent>>`) | Rol (`<<role>>`) | Arquitectura | Motor de Machine Learning | Relacion `<control>` |
| :--- | :--- | :--- | :--- | :--- |
| `AgenteCoordinador` | `Coordinador` | Basado en utilidad | Reglas de utilidad sobre margenes de BD | Supervisa a los otros 4 roles |
| `AgenteVentas` | `EjecutivoVentas` | Con aprendizaje + Utilidad | NLP (`TfidfVectorizer` + `LogisticRegression`) + catalogo de BD | Subordinado a `Coordinador` |
| `AgenteCotizaciones` | `LiquidadorCotizaciones` | Con aprendizaje + Metas | `RandomForestRegressor` (Scikit-Learn) | Subordinado a `Coordinador` |
| `AgentePromociones` | `AnalistaPromociones` | Con aprendizaje + Utilidad | `KNeighborsRegressor` (KNN Scikit-Learn) | Subordinado a `Coordinador` |
| `AgenteAgendamientos` | `GestorAgenda` | Basado en metas | Consulta y reserva sobre `agendaVirtual` en BD | Subordinado a `Coordinador` |

---

## 3. Configuracion de la Base de Datos en la Nube (Supabase)

1. Crea un proyecto gratuito en `https://supabase.com/`.
2. Abre **SQL Editor -> New query** en tu proyecto de Supabase, pega el contenido completo de [`sql/supabase_schema_y_seed.sql`](sql/supabase_schema_y_seed.sql) y presiona **Run**.
   - El script crea las 8 tablas (`negocios`, `servicios`, `promociones_vigentes`, `tarifas_historicas`, `historico_ofertas`, `agenda_bloques`, `perfiles_clientes`, `alertas_humanas`) y carga dos negocios independientes de ejemplo (`tech_freelance` y `estudio_fotografia`).
3. Ve a **Project Settings -> API** y copia:
   - **Project URL** -> `SUPABASE_URL`
   - **service_role (secret) key** -> `SUPABASE_KEY`
4. Copia `.env.example` como `.env` y completa:
   ```env
   SUPABASE_URL=https://tu-proyecto.supabase.co
   SUPABASE_KEY=tu_service_role_key
   NEGOCIO_ID=tech_freelance
   ```
   *(Si cambias `NEGOCIO_ID=estudio_fotografia` o creas una fila nueva en `negocios` y `servicios`, todo el sistema multiagente cambia de rubro automaticamente sin editar codigo).*

---

## 4. Ejecucion y Conexion con Botpress

```bash
pip install -r requirements.txt
python main.py
```

Esto inicia:
- Servidor XMPP embebido (`localhost:5222`) con los 5 agentes SPADE conectados.
- API puente para Botpress en `http://127.0.0.1:8000/api/botpress/mensaje`.
- Monitor Web FIPA-ACL y Supervisión Human-in-the-Loop en `http://127.0.0.1:8000/`.

Para conectar Botpress Cloud con tu PC local, expone el puerto `8000`:

```bash
ngrok http 8000
```

Cuando un cliente escala a atencion humana (`notificar_dueno: true`), Botpress notifica al WhatsApp personal del dueno (`telefono_dueno` / `OWNER_WHATSAPP_NUMBER`) con los datos del cliente y pausa la automatizacion unicamente en ese chat (`responder_al_cliente: false` en mensajes posteriores hasta reactivar en el monitor).

---

## 5. Pruebas Automatizadas

```bash
pytest -v
```
