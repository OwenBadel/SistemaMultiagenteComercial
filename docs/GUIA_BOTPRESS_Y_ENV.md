# Guía de Integración Botpress + Variables `.env` (Plataforma Multi-Negocio)

Esta guía resume cómo configurar tu archivo `.env` y conectar uno o varios bots de **Botpress Studio** (por ejemplo, **Mile Cuentas Beauty** y **Estudio Visual Independiente**) al servidor multi-negocio actual en la rama `main`.

> **Sobre las ramas del repositorio:**
> - `main`: plataforma comercial multi-negocio con **Google Gemini Flash** + Panel PWA.
> - `v1-universidad`: respaldo intacto de la versión académica anterior (SPADE + Ollama + Scikit-Learn).

---

## 1. Qué va en el archivo `.env`

Copia `.env.example` como `.env` en la raíz del proyecto y completa solo estas variables:

```env
# --- MOTOR IA (GOOGLE AI STUDIO) ---
GEMINI_API_KEY=tu_api_key_de_google_ai_studio
GEMINI_MODEL=gemini-3.1-flash-lite

# --- BASE DE DATOS MULTI-NEGOCIO (SUPABASE O SQLITE) ---
SUPABASE_URL=https://tu-proyecto.supabase.co/rest/v1/
SUPABASE_KEY=tu_secret_key_o_service_role
SQLITE_DB_PATH=data/comercial_multitenant.db
# Solo respaldo si alguien llama al endpoint antiguo /api/botpress/mensaje sin indicar negocio:
NEGOCIO_ID=estudio_fotografia

# --- GOOGLE CALENDAR API v3 (CADA NEGOCIO USA SU PROPIO CALENDAR_ID EN SUPABASE) ---
GOOGLE_SERVICE_ACCOUNT_JSON=google_service_account.json
GOOGLE_TIMEZONE=America/Bogota

# --- SEGURIDAD DEL PANEL PWA ---
ADMIN_ACCESS_KEY=cambia_esta_clave_admin
SESSION_SECRET=cambia_este_secreto_largo_y_aleatorio
PUBLIC_BASE_URL=https://tu-url-publica-o-ngrok.app

# --- CONEXION WHATSAPP (BOTPRESS Y/O META CLOUD API) ---
BOTPRESS_API_KEY=una_clave_secreta_para_botpress
WHATSAPP_VERIFY_TOKEN=sma_whatsapp_verify_2026
WHATSAPP_APP_SECRET=
WHATSAPP_ACCESS_TOKEN=
# Solo poner en true para pruebas locales sin llaves de webhook (NUNCA en produccion)
ALLOW_INSECURE_WEBHOOKS=false

# --- SERVIDOR HTTP ---
HOST=0.0.0.0
PORT=8000
```

### Variables antiguas que YA NO se usan (puedes borrarlas si las tenías)
- `OLLAMA_HOST`, `LLAMA_MODEL`, `LLAMA_TIMEOUT_SECONDS`, `LLAMA_AUTO_INSTALL` (reemplazados por `GEMINI_API_KEY` y `GEMINI_MODEL`).
- `OWNER_WHATSAPP_NUMBER` (ahora cada negocio guarda su propio `telefono_dueno` en Supabase y se edita desde el Panel PWA).
- `HUMAN_REACTIVATION_API_KEY` (reemplazada por `ADMIN_ACCESS_KEY` y PIN individual por dueño).

> **Sobre `NEGOCIO_ID`:** ya **no** limita el servidor a un solo negocio. Cuando entra un mensaje por `/api/webhook/{negocio_id}` o por `/api/whatsapp/webhook`, ese valor de `.env` se ignora. Solo actúa como respaldo si alguien llama a `/api/botpress/mensaje` sin especificar negocio.

---

## 2. Arranque del servidor y túnel HTTPS (`ngrok`)

En una terminal dentro del proyecto:

```bash
pip install -r requirements.txt
python main.py
```

Para que Botpress Cloud alcance tu servidor local (si aún no está en un VPS/nube):

```bash
ngrok http 8000
```

Copia la URL HTTPS que entrega `ngrok` (ej. `https://abc123.ngrok-free.app`) y pégala también en `PUBLIC_BASE_URL` dentro de `.env` para que los enlaces al panel que llegan en las alertas funcionen desde cualquier celular.

---

## 3. Cómo conectar múltiples negocios en Botpress con UN solo link

Un mismo servidor (`python main.py`) y un mismo túnel `ngrok` atienden a todos los negocios al mismo tiempo. Solo cambia la parte final de la URL (`{negocio_id}`):

- **Mile Cuentas Beauty:** `POST https://TU_URL_HTTPS/api/webhook/mile_cuentas_beauty`
- **Estudio Visual Independiente:** `POST https://TU_URL_HTTPS/api/webhook/estudio_fotografia`
- **Cualquier negocio nuevo que crees en el panel:** `POST https://TU_URL_HTTPS/api/webhook/id_del_negocio`

### Paso a paso en Botpress Studio (por cada bot/negocio)
1. En Botpress Studio, crea un bot para el negocio y conéctale su integración de **WhatsApp** con el número dedicado de ese negocio.
2. En el flujo principal (tarjeta **Execute Code** o petición HTTP), envía un `POST` a la URL del negocio correspondiente:
   - **Headers obligatorios:**
     ```text
     Content-Type: application/json
     X-Botpress-Key: <el mismo BOTPRESS_API_KEY de tu .env>
     ```
     *(Nota: si `BOTPRESS_API_KEY` está vacío y `ALLOW_INSECURE_WEBHOOKS=false`, el servidor responde `503` por seguridad).*
   - **Body JSON:**
     ```json
     {
       "id_cliente": "{{event.tags.conversation['whatsapp:userId'] || event.userId}}",
       "nombre": "{{user.name || 'Cliente'}}",
       "texto": "{{event.preview}}",
       "canal": "WhatsApp"
     }
     ```
3. **Procesa la respuesta JSON del servidor en Botpress:**
   - `responder_al_cliente` (`true`/`false`): si es `true`, envía `respuesta` al usuario por WhatsApp. Si es `false` (`pausado_humano: true`), **no envíes nada** (el chat está en silencio porque lo está atendiendo el dueño).
   - `notificar_dueno` (`true`/`false`): cuando es `true`, el chat acaba de escalarse a atención humana. Usa `telefono_dueno` y `mensaje_notificacion_dueno` para avisar al dueño (o el dueño lo atiende directo desde el Panel PWA).

---

## 4. Número del negocio vs. número personal de alertas

En el **Panel PWA** (`http://127.0.0.1:8000/` -> entra como **Super-Admin** -> elige el negocio -> **Reglas y Bot**):

- **`WhatsApp del negocio`:** el número conectado al bot donde escriben los clientes.
- **`WhatsApp personal del dueño (alertas)`:** el celular personal del dueño donde quiere recibir avisos de escalamiento.

> **Importante:** En la API oficial de WhatsApp (vía Botpress o Meta directo), un bot no puede enviarse alertas por WhatsApp a su propio número ni mantener la app estándar de WhatsApp abierta en la misma línea (salvo líneas habilitadas con *Coexistence*). Por eso se recomienda usar un número dedicado para la automatización del negocio y el número personal del dueño para recibir alertas y entrar al Panel PWA.
