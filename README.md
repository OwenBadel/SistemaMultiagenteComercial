# SistemaMultiagenteComercial

Sistema multiagente comercial en Python construido con **SPADE**, **Meta Llama local (Ollama)**, **Scikit-Learn**, **FastAPI**, **Supabase**, **Google Calendar API v3** y **Botpress**. Disenado para operar de forma generica con cualquier trabajador independiente o negocio leyendo su catalogo, reglas comerciales, precios historicos y configuracion directamente desde base de datos en la nube.

---

## 1. Como Funciona el Proyecto

```text
Cliente (WhatsApp / Webchat / Instagram / Messenger)
  -> Botpress (pasarela de canales y alertas salientes al dueno)
  -> POST /api/botpress/mensaje (FastAPI)
  -> ServicioMultiagenteComercial (5 agentes SPADE sobre XMPP embebido)
       |-- EntornoVirtualAtencion <-> Supabase PostgreSQL (negocio, catalogo, ML, perfiles, citas y alertas)
       |-- EntornoVirtualAtencion <-> Google Calendar API v3 (disponibilidad y citas reales)
  <- JSON { respuesta, responder_al_cliente, pausado_humano, notificar_dueno, ... }
```

1. **Atencion multicanal via Botpress:** Todo mensaje entrante de un cliente llega al endpoint `POST /api/botpress/mensaje`.
2. **Negocio 100% definido por base de datos:** Al iniciar, el sistema carga desde Supabase (`NEGOCIO_ID`) el rubro, moneda, unidad de cobro (`horas`, `sesiones`, etc.), catalogo de servicios, palabras clave, margenes minimos y topes de descuento.
3. **Procesamiento colaborativo en SPADE + Llama Local + Scikit-Learn:** `AgenteVentas` utiliza un modelo generativo local **Meta Llama** (vía Ollama con salidas estructuradas JSON Schema) para comprender la intención, tono y parámetros del cliente y redactar respuestas naturales sin depender de claves en la nube ni cuotas HTTP 429; `AgenteCotizaciones` liquida presupuestos (tarifario por sesiones, motor de economía unitaria anti-pérdida y `RandomForestRegressor` para proyectos por horas); `AgentePromociones` calcula incentivos con `KNeighborsRegressor`; `AgenteAgendamientos` consulta/reserva citas reales en Google Calendar + Supabase; y `AgenteCoordinador` ejecuta control **Human-in-the-Loop** por chat cuando detecta un caso crítico.
4. **Control Human-in-the-Loop aislado por chat:** Si un cliente entra en atencion humana, el sistema pausa unicamente ese `id_cliente`, marca `notificar_dueno: true` para que Botpress avise al WhatsApp personal del dueno, y devuelve `responder_al_cliente: false` en mensajes posteriores mientras el dueno responde desde su WhatsApp corporativo. Desde el monitor web (`GET /`) se puede reactivar la automatizacion al terminar.

---

## 2. Agentes, Capacidades, Arquitectura y Modelos de IA

| Agente | Rol | Que es capaz de hacer | Arquitectura | Motor de IA / Machine Learning |
| :--- | :--- | :--- | :--- | :--- |
| `AgenteVentas` | `EjecutivoVentas` | Unica puerta conversacional con el cliente. Interpreta mensajes en contexto, detecta tono (`interesado`, `indeciso`, `hostil`), extrae parametros dinamicamente segun el catalogo de BD, pre-perfila datos faltantes, envia el portafolio y redacta respuestas naturales con validador anti-alucinacion. | Generativo local (LLM Llama / Ollama) y basado en utilidad | **Meta Llama Local via Ollama** (`llama3.1:8b` en GPU >= 6 GB VRAM o `llama3.2:3b` en CPU) con **Structured Outputs (JSON Schema)** para NLU + redaccion NLG local (auto-instalacion automatica en arranque) |
| `AgenteCoordinador` | `Coordinador` | Orquesta turnos entre especialistas, enruta peticiones, audita propuestas comerciales y descuentos frente al margen minimo anti-perdida configurado en BD y suspende la automatizacion de un chat escalandolo al dueno ante casos criticos. | Basado en utilidad | Evaluacion racional de utilidad y piso minimo de rentabilidad sobre margenes configurados en BD |
| `AgenteCotizaciones` | `LiquidadorCotizaciones` | Valida requisitos obligatorios segun el servicio solicitado (pidiendo aclaraciones si faltan datos) y calcula cotizaciones formales con desglose economico usando tarifa por sesiones, economia unitaria anti-perdida (solicitudes a medida/fraccionadas) o el historico del negocio. | Con aprendizaje y basado en metas | **RandomForestRegressor** (Scikit-Learn) entrenado con `tarifas_historicas` (servicios por horas) + tarifa configurada y motor unitario anti-perdida (sesiones) |
| `AgentePromociones` | `AnalistaPromociones` | Evalua promociones vigentes y propone el porcentaje optimo de descuento ante objeciones de precio, equilibrando probabilidad de cierre y margen de ganancia. | Con aprendizaje y basado en utilidad | **KNeighborsRegressor (KNN)** con `StandardScaler` (Scikit-Learn) sobre `historico_ofertas` |
| `AgenteAgendamientos` | `GestorAgenda` | Verifica disponibilidad real en la agenda del dueno, propone horarios libres alternativos del dia si el bloque esta ocupado y crea la cita confirmada en calendario. | Basado en metas | Integracion directa con **Google Calendar API v3** (`freeBusy` y `events.insert`) |

---

## 3. Configuracion (Supabase, Google Calendar y Botpress)

### Base de datos en la nube (Supabase)
1. Crea un proyecto en `https://supabase.com/`.
2. En **SQL Editor -> New query**, pega y ejecuta [`sql/supabase_schema_y_seed.sql`](sql/supabase_schema_y_seed.sql) (crea las 8 tablas: `negocios`, `servicios`, `promociones_vigentes`, `tarifas_historicas`, `historico_ofertas`, `perfiles_clientes`, `alertas_humanas` y `agendamientos`, e incluye los negocios de ejemplo `tech_freelance` y `estudio_fotografia`).
3. En **Project Settings -> Data API / API Keys**, copia tu **Project URL** (`https://...supabase.co`) y tu **Secret key** (`sb_secret_...`).

### Agenda real (Google Calendar API v3)
1. En **Google Cloud Console**, habilita **Google Calendar API**, crea una **Service Account** y descarga su archivo JSON (ej. `google_service_account.json` en la raiz del proyecto).
2. En **Google Calendar**, comparte el calendario del dueno con el `client_email` del JSON otorgando permiso **"Hacer cambios en eventos"**.

### Variables de entorno (`.env`)
Copia `.env.example` como `.env` y completa tus credenciales:

```env
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu_secret_key
NEGOCIO_ID=tech_freelance

GOOGLE_CALENDAR_ID=tu_correo@gmail.com
GOOGLE_SERVICE_ACCOUNT_JSON=google_service_account.json
GOOGLE_TIMEZONE=America/Bogota

OWNER_WHATSAPP_NUMBER=573001234567
BOTPRESS_API_KEY=
HUMAN_REACTIVATION_API_KEY=admin_sma_2026
OLLAMA_HOST=http://127.0.0.1:11434
LLAMA_MODEL=auto
LLAMA_AUTO_INSTALL=true
HOST=0.0.0.0
PORT=8000
```

Las reservas se guardan en `public.agendamientos`. Sin Google Calendar configurado, la cita se persiste en Supabase con `registrado_en='solo_supabase'` y `google_calendar_estado='no_configurado'`. Si Google está configurado pero no se puede validar su disponibilidad, la solicitud queda persistida como `pendiente` y no se confirma al cliente. Si Google valida el horario pero falla al crear el evento, se confirma solo después de guardar la cita en Supabase y queda registrada con `google_calendar_estado='error'`.

Si no configuras `SUPABASE_URL` y `SUPABASE_KEY`, el proyecto usa SQLite local persistente para desarrollo. En ese modo, las citas se conservan en SQLite; para producción y almacenamiento en `public.agendamientos`, configura Supabase.

Al solicitar una cita se entienden fechas relativas como “hoy”, “mañana”, “pasado mañana”, “ayer”, “en/dentro de N días”, expresiones de día del mes (“el 30 a las 3pm”) y horarios como “mediodía”, calculados en `America/Bogota`. Si solo se indica el día sin hora, el agente conserva la fecha parcial y pregunta explícitamente la hora deseada en vez de asumir una por defecto; las fechas pasadas no se reservan y se solicita una fecha futura. Si no se reconoce ninguna fecha, el agente pide aclaración en vez de asignar una fecha por defecto.

Si Google Calendar crea un evento, pero Supabase no puede confirmar el estado sincronizado, la cita queda pendiente con el identificador del evento para reconciliación manual; el agente no confirma al cliente ni elimina automáticamente el evento. Las citas pendientes bloquean el horario, así que no se debe reintentar la sincronización hasta revisar el registro y ese ID en ambas fuentes. Si tampoco se puede guardar el marcador de recuperación en Supabase, el ID del evento se deja en el log para revisión manual.

---

## 4. Instalacion, Ejecucion y Pruebas

```bash
pip install -r requirements.txt
python main.py
```

Servicios disponibles al arrancar:
- Monitor de Telemetria en Tiempo Real y Supervisión Human-in-the-Loop: `http://127.0.0.1:8000/`
- Endpoint para Botpress Studio: `POST http://127.0.0.1:8000/api/botpress/mensaje` (exponible con `ngrok http 8000`)

Pruebas automatizadas:

```bash
pytest -v
```

---

## @Authors

- **Owen Badel**
- **Maverick Diaz**
- **Jesus Viloria**
