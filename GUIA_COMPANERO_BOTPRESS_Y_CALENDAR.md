# Guia de Integracion: Botpress y Google Calendar

Este documento explica paso a paso que falta configurar en plataformas externas (**Google Cloud**, **Google Calendar** y **Botpress Studio**) y que valores reemplazar para conectar todo con el backend de `SistemaMultiagenteComercial`.

El codigo en Python (los 5 agentes SPADE, modelos Scikit-Learn, conexion a Supabase y servidor FastAPI) ya esta completo y probado. Solo falta conectar las cuentas externas.

---

## 1. Configuracion en Google Cloud y Google Calendar (Agenda Real)

Para que `AgenteAgendamientos` consulte disponibilidad real y cree citas directamente en el calendario del dueno del negocio:

### Paso 1.1: Google Cloud Console (`https://console.cloud.google.com/`)
1. Inicia sesion con una cuenta Google y crea o selecciona un proyecto.
2. Ve a **APIs y servicios -> Biblioteca**, busca **Google Calendar API** y haz clic en **Habilitar**.
3. Ve a **APIs y servicios -> Credenciales -> Crear credenciales -> Cuenta de servicio (Service Account)**.
   - Ponle un nombre (ej. `bot-agenda-comercial`) y finaliza la creacion.
4. Entra a la cuenta de servicio creada, ve a la pestana **Claves -> Agregar clave -> Crear clave nueva -> JSON**.
5. Se descargara un archivo `.json`. Muevelo a la raiz del proyecto (`SistemaMultiagenteComercial/`) y renombralo exactamente como:
   ```text
   google_service_account.json
   ```
   *(Este archivo esta ignorado en `.gitignore` para no exponer llaves en GitHub).*

### Paso 1.2: Compartir el Google Calendar (`https://calendar.google.com/`)
1. Abre `google_service_account.json` con cualquier editor y copia el correo que aparece en el campo `"client_email"` (ejemplo: `bot-agenda-comercial@tu-proyecto.iam.gserviceaccount.com`).
2. Abre **Google Calendar** con la cuenta del dueno donde se registraran las citas.
3. En la columna izquierda (**Mis calendarios**), pasa el mouse sobre tu calendario -> clic en los tres puntos -> **Configuracion y uso compartido**.
4. Baja hasta **Compartir con determinadas personas o grupos** -> **Agregar personas y grupos**.
5. Pega el `client_email` de la Service Account y en permisos selecciona **"Hacer cambios en eventos"** (obligatorio para poder crear citas).

### Paso 1.3: Completar el archivo `.env` del proyecto
Abre el archivo `.env` en la raiz del proyecto y rellena/verifica estas variables:

```env
# --- BASE DE DATOS EN LA NUBE (SUPABASE) ---
SUPABASE_URL=https://REEMPLAZAR_CON_TU_PROJECT_URL.supabase.co
SUPABASE_KEY=REEMPLAZAR_CON_TU_SECRET_KEY_DE_SUPABASE
NEGOCIO_ID=estudio_fotografia

# --- GOOGLE CALENDAR API v3 ---
# Reemplaza con el correo del calendario que compartiste en el Paso 1.2
GOOGLE_CALENDAR_ID=REEMPLAZAR_CON_CORREO_DEL_CALENDARIO@gmail.com
GOOGLE_SERVICE_ACCOUNT_JSON=google_service_account.json
GOOGLE_TIMEZONE=America/Bogota

# --- NOTIFICACION AL DUENO (WHATSAPP PERSONAL) ---
# Formato internacional sin + ni espacios (ejemplo Colombia: 573001234567)
OWNER_WHATSAPP_NUMBER=57REEMPLAZAR_CON_NUMERO_PERSONAL_DEL_DUENO

# --- SEGURIDAD OPCIONAL Y PUERTO ---
BOTPRESS_API_KEY=
HOST=0.0.0.0
PORT=8000
```

> **Nota sobre la tabla `agenda_bloques` en Supabase:** Si ejecutaste anteriormente una version previa del SQL que incluia `agenda_bloques`, entra al **SQL Editor** de Supabase y ejecuta una sola vez:
> ```sql
> DROP TABLE IF EXISTS agenda_bloques;
> ```

---

## 2. Arrancar el Backend Local y Exponerlo por HTTPS

En la maquina donde correra el servidor Python:

1. Instala dependencias (si no lo has hecho):
   ```bash
   pip install -r requirements.txt
   ```
2. Inicia el sistema multiagente y la API HTTP:
   ```bash
   python main.py
   ```
   En consola debe mostrarse:
   - `Base de Datos`: `Supabase Cloud (PostgreSQL)`
   - `Agenda`: `Google Calendar API v3`
   - Monitor Web disponible en `http://127.0.0.1:8000/`
3. En una **segunda terminal**, expone el puerto `8000` hacia internet para que Botpress Cloud pueda consultarlo:
   ```bash
   ngrok http 8000
   ```
   Copia la URL HTTPS publica generada por ngrok (ejemplo: `https://1a2b-3c4d.ngrok-free.app`).

---

## 3. Configuracion en Botpress Studio (`https://studio.botpress.cloud/`)

En tu bot de Botpress (donde ya estan conectados WhatsApp corporativo, Webchat, Instagram o Messenger), Botpress actua como la pasarela conversacional de primera linea hacia nuestra API en Python:

### Paso 3.1: Modelo de Responsabilidad (Opcion A: Botpress NLU + Python TF-IDF)
- **Rol de Botpress:** Captura el mensaje en los canales reales (WhatsApp, Webchat, etc.), gestiona los turnos de conversacion y entrega el texto directamente al backend. De forma opcional, si deseas que el NLU de Botpress identifique la intencion previamente, puede enviar el campo `tipo_evento_sugerido`.
- **Rol del Backend Python:** El `AgenteVentas` analiza el texto con su modelo Scikit-Learn TF-IDF enriquecido y reglas lexicas, o procesa directamente la sugerencia de Botpress. Los agentes resuelven la cotizacion (Random Forest), el descuento (KNN), la cita en Google Calendar y la respuesta comercial final en milisegundos.
- **Importante:** Desactiva respuestas automaticas de **Knowledge Base** o nodos **Autonomous** libres que respondan por su cuenta antes de ejecutar el codigo de enlace, para que sea siempre el sistema multiagente quien genere las respuestas de negocio.

### Paso 3.2: Crear Variables de Flujo (Workflow Variables)
En el flujo principal de Botpress Studio, crea estas 5 variables en la seccion **Variables**:
- `respuestaSMA` (tipo **String**)
- `responderAlCliente` (tipo **Boolean**)
- `notificarDueno` (tipo **Boolean**)
- `telefonoDueno` (tipo **String**)
- `mensajeNotificacionDueno` (tipo **String**)

### Paso 3.3: Configurar la tarjeta `Execute Code`
Justo despues de recibir el mensaje del usuario (por ejemplo, tras una tarjeta **Wait for User Input** o en el nodo inicial), agrega una tarjeta **Execute Code** y pega este codigo reemplazando **`REEMPLAZA_AQUI_TU_URL_NGROK`** por tu URL HTTPS de ngrok:

```javascript
const URL_SMA = 'https://REEMPLAZA_AQUI_TU_URL_NGROK/api/botpress/mensaje'

// Si configuraste BOTPRESS_API_KEY en .env, pon el mismo valor aqui; si no, dejalo vacio ''
const API_KEY = ''

try {
  const headers = API_KEY ? { 'x-api-key': API_KEY } : {}

  const res = await axios.post(
    URL_SMA,
    {
      channel: event.channel || 'WhatsApp',
      userId: event.userId || event.conversationId,
      userName: event.tags?.user?.name || 'Cliente',
      text: event.preview,
      // Opcional: si Botpress clasifica la intencion con su NLU/IA, puede sugerirla:
      // tipo_evento_sugerido: event.nlu?.intent?.name || null
    },
    { headers, timeout: 15000 }
  )

  workflow.respuestaSMA = res.data.respuesta || ''
  workflow.responderAlCliente = Boolean(res.data.responder_al_cliente)
  workflow.notificarDueno = Boolean(res.data.notificar_dueno)
  workflow.telefonoDueno = res.data.telefono_dueno || ''
  workflow.mensajeNotificacionDueno = res.data.mensaje_notificacion_dueno || ''
} catch (err) {
  workflow.respuestaSMA = 'En este momento estamos verificando tu solicitud, intenta nuevamente en un momento.'
  workflow.responderAlCliente = true
  workflow.notificarDueno = false
}
```

### Paso 3.4: Conectar las condiciones de salida (Transiciones)
Despues de la tarjeta **Execute Code**, configura las salidas del nodo usando tarjetas **Expression / Transition**:

1. **Rama 1 — Notificar al dueno (si `workflow.notificarDueno === true`):**
   - Condicion: `workflow.notificarDueno === true`
   - Accion: Conecta la tarjeta de la integracion de WhatsApp (**Start Conversation** / **Send Message**) enviando al telefono **`{{workflow.telefonoDueno}}`** el mensaje **`{{workflow.mensajeNotificacionDueno}}`**.
   - Luego continua hacia la verificacion de `workflow.responderAlCliente` para despedir al cliente una unica vez antes de silenciar el bot en ese chat.
   - *(Importante sobre WhatsApp Cloud API: si pasaron mas de 24 horas desde que el numero personal del dueno escribio al numero corporativo, WhatsApp exige usar una plantilla aprobada en Meta con las variables del mensaje).*

2. **Rama 2 — Responder al cliente (si `workflow.responderAlCliente === true`):**
   - Condicion: `workflow.responderAlCliente === true`
   - Accion: Tarjeta **Text** normal mostrando `{{workflow.respuestaSMA}}`.
   - Despues vuelve a esperar el siguiente mensaje del usuario (**Wait for User Input**).

3. **Rama 3 — Silencio Human-in-the-Loop (si `workflow.responderAlCliente === false`):**
   - Condicion: `workflow.responderAlCliente === false`
   - Accion: No enviar ningun mensaje de texto al cliente. Simplemente vuelve a esperar entrada (**Wait for User Input**). Mientras tanto, el dueno conversa directamente desde el WhatsApp corporativo con ese cliente, mientras otros clientes siguen recibiendo respuestas automaticas normales.

---

## 4. Checklist de Prueba Final

1. Abre el monitor web en `http://127.0.0.1:8000/` y verifica arriba las etiquetas:
   - `BD: Supabase Cloud (PostgreSQL)`
   - `Agenda: Google Calendar API v3`
   - `Negocio` cargado correctamente
2. Escribe desde WhatsApp o Webchat de Botpress:
   - Consulta inicial: *"Hola, que servicios ofrecen?"* -> debe listar los servicios reales de Supabase.
   - Cotizacion: *"Quiero cotizar 3 sesiones de retrato corporativo"* (si usas `estudio_fotografia`) o *"Cotizar desarrollo web de 40 horas complejidad 3"* (si usas `tech_freelance`) -> debe liquidar con `RandomForestRegressor`.
   - Negociacion: *"Esta muy costoso para mi presupuesto"* -> debe proponer descuento optimo con `KNeighborsRegressor`.
   - Agenda: *"Quiero agendar cita para el 2026-11-20 09:00"* -> debe aparecer el evento creado en el **Google Calendar** real del dueno.
   - Escalamiento: *"Esto es una estafa, necesito hablar con el dueno ya"* -> debe llegar la notificacion al WhatsApp personal del dueno, quedar silenciado solo ese chat y mostrarse la alerta en `http://127.0.0.1:8000/` con el boton **Reactivar bot**.
