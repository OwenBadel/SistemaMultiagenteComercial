# SistemaMultiagenteComercial — Plataforma Comercial Multi-Negocio

Plataforma SaaS comercial multi-negocio para atención automatizada por WhatsApp, cotización determinista, agendamiento real en Google Calendar y supervisión humana desde un panel móvil instalable (PWA). Construida en Python con **FastAPI**, **Google Gemini Flash** (`google-genai` con Tool Calling nativo, modelo recomendado `gemini-3.1-flash-lite`), **Supabase PostgreSQL / SQLite**, **Google Calendar API v3** y compatibilidad dual con **Botpress Studio** y **Meta WhatsApp Cloud API**.

> **Nota sobre ramas:** La rama `v1-universidad` conserva intacta la arquitectura académica original (SPADE + Llama local + Scikit-Learn). Esta rama `main` implementa la plataforma comercial multi-negocio sobre un único servidor.
>
> **Guía rápida para Botpress y `.env`:** consulta [`docs/GUIA_BOTPRESS_Y_ENV.md`](docs/GUIA_BOTPRESS_Y_ENV.md).

---

## 1. Arquitectura General

```text
Clientes WhatsApp (Negocio A: Fotografia, Negocio B: Belleza, Negocio N: Pizzeria...)
  |
  +--> Botpress Studio:           POST /api/webhook/{negocio_id}  (o POST /api/botpress/mensaje)
  +--> Meta WhatsApp Cloud API:   GET/POST /api/whatsapp/webhook  (enrutamiento automatico por numero)
  |
  v
FastAPI (src/comercial/servidor_comercial.py)
  |-- Aislamiento estricto por negocio_id (reglas, catalogo, horarios, agenda, chats y dueno)
  |-- OrquestadorComercialGemini (Gemini Flash + 5 herramientas deterministas)
  |     1. consultar_servicios
  |     2. calcular_cotizacion
  |     3. consultar_horarios_disponibles (horarios por dia + Google Calendar + BD)
  |     4. agendar_cita (bloqueo anti-traslape + Google Calendar API v3)
  |     5. escalar_a_humano (pausa silenciosa por chat + alerta WhatsApp al dueno)
  |-- Panel Web Movil PWA (GET /)
        |-- Rol Dueno (acceso con PIN personal, solo ve su negocio)
        |-- Rol Administrador (gestion total + asistente conversacional Gemini para alta de negocios)
```

### Capacidades principales

1. **Multi-negocio en una sola instancia:** Dos o más negocios (`estudio_fotografia`, `salon_belleza` o cualquier rubro nuevo) operan simultáneamente con separación total de números de WhatsApp, dueños, catálogos, horarios laborales por día, promociones, reglas de atención, calendarios y chats.
2. **Motor Gemini Flash con herramientas deterministas:** El modelo comprende lenguaje natural libre sin depender de palabras clave rígidas, mientras todos los precios, cálculos, horarios y reservas se ejecutan mediante herramientas deterministas conectadas a la base de datos.
3. **Escalamiento humano por chat con notificación al dueño:** Cuando un cliente requiere atención personal (reclamos, pedidos fuera de catálogo o solicitud explícita de hablar con una persona), se pausa únicamente ese chat, se envía una alerta al WhatsApp personal del dueño con un resumen breve y enlace directo `wa.me`, y el caso aparece en la bandeja del panel PWA para reactivar el asistente cuando corresponda.
4. **Alta de nuevos negocios sin tocar código:** Desde el panel de Administrador, un asistente conversacional guiado por Gemini entrevista al operador o al dueño (rubro, servicios, precios, duración, horarios, promociones y números de WhatsApp), rellena automáticamente un formulario editable y crea el negocio listo para operar.
5. **Panel PWA móvil instalable:** Diseñado con enfoque mobile-first, instalable en Android y iOS desde el navegador, con simulador de chat integrado por negocio.

---

## 2. Base de Datos y Configuración

### Opción A: Supabase PostgreSQL (recomendada para producción)
1. **Si tu proyecto Supabase es nuevo:** pega y ejecuta únicamente [`sql/supabase_schema_y_seed.sql`](sql/supabase_schema_y_seed.sql) en el **SQL Editor** de Supabase (crea las 6 tablas con RLS, índices únicos por canal y carga `estudio_fotografia` y `mile_cuentas_beauty`).
2. **Si ya tenías el esquema antiguo de la versión universitaria:** ejecuta la migración aditiva [`sql/supabase_comercial_v2.sql`](sql/supabase_comercial_v2.sql).

### Opción B: SQLite local (desarrollo inmediato)
Si dejas `SUPABASE_URL` y `SUPABASE_KEY` vacías en `.env`, el servidor inicializa automáticamente `data/comercial_multitenant.db` con `estudio_fotografia` y `salon_belleza` precargados.

### Variables de entorno (`.env`)
Copia `.env.example` como `.env` y completa tus claves:

```env
GEMINI_API_KEY=tu_api_key_de_google_ai_studio
GEMINI_MODEL=gemini-3.8-flash

SUPABASE_URL=
SUPABASE_KEY=
SQLITE_DB_PATH=data/comercial_multitenant.db
NEGOCIO_ID=estudio_fotografia

GOOGLE_SERVICE_ACCOUNT_JSON=google_service_account.json
GOOGLE_TIMEZONE=America/Bogota

ADMIN_ACCESS_KEY=reemplaza_con_clave_admin_fuerte
SESSION_SECRET=reemplaza_con_secreto_largo_aleatorio
PUBLIC_BASE_URL=https://tu-dominio.com

BOTPRESS_API_KEY=reemplaza_con_api_key_webhook
WHATSAPP_VERIFY_TOKEN=sma_whatsapp_verify_2026
WHATSAPP_APP_SECRET=reemplaza_con_app_secret_de_meta
WHATSAPP_ACCESS_TOKEN=
ALLOW_INSECURE_WEBHOOKS=false

HOST=0.0.0.0
PORT=8000
```

> **Seguridad en producción:**
> - El PIN inicial precargado para dueños de ejemplo es `1234` (tanto en SQLite como en `sql/supabase_comercial_v2.sql`). Cámbialo desde la pestaña **Reglas y Bot** antes de salir a producción (al cambiarlo se revocan automáticamente sesiones anteriores).
> - Los webhooks externos operan en modo **fail-closed**: si `BOTPRESS_API_KEY` o `WHATSAPP_APP_SECRET` están vacías, responden `503` salvo que actives `ALLOW_INSECURE_WEBHOOKS=true` exclusivamente para pruebas locales.

---

## 3. Ejecución y Pruebas

```bash
pip install -r requirements.txt
python main.py
```

Accesos al iniciar el servidor:
- **Panel PWA (Dueños y Administrador):** `http://127.0.0.1:8000/`
- **Webhook Botpress por negocio:** `POST http://127.0.0.1:8000/api/webhook/{negocio_id}`
- **Webhook Botpress compatible:** `POST http://127.0.0.1:8000/api/botpress/mensaje`
- **Webhook Meta WhatsApp Cloud API:** `GET/POST http://127.0.0.1:8000/api/whatsapp/webhook`
- **Salud del servicio:** `GET http://127.0.0.1:8000/health`

Ejecución de pruebas automatizadas de la plataforma comercial:

```bash
pytest tests/test_comercial_multitenant.py -v
```

---

## 4. Integración con WhatsApp

### Con Botpress Studio
- Crea un bot por negocio (o un flujo enrutado) apuntando al webhook `POST https://tu-servidor/api/webhook/{negocio_id}`.
- Envía tu clave `BOTPRESS_API_KEY` mediante el encabezado `X-Botpress-Key` (también se acepta `X-Api-Key` o `Authorization: Bearer`).
- El servidor responde `{ respuesta, responder_al_cliente, pausado_humano, notificar_dueno, telefono_dueno, mensaje_notificacion_dueno, enlace_whatsapp_cliente, enlace_panel_dueno }`.

### Con Meta WhatsApp Cloud API directa
- Configura el webhook en Meta for Developers hacia `https://tu-servidor/api/whatsapp/webhook` usando `WHATSAPP_VERIFY_TOKEN` y `WHATSAPP_APP_SECRET` (validación obligatoria `X-Hub-Signature-256`).
- Registra desde el panel de Super-Admin el `whatsapp_phone_number_id` de cada negocio (y opcionalmente su `whatsapp_access_token` propio o usa el global `WHATSAPP_ACCESS_TOKEN`).
- El servidor identifica automáticamente a qué negocio pertenece cada mensaje entrante, responde al cliente vía Graph API v21.0 y, ante un escalamiento, envía la alerta directa al WhatsApp personal del dueño.

---

## @Authors

- **Owen Badel Hooker**
- **Maverick Diaz**
- **Jesus Viloria**
