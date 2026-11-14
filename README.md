# SistemaMultiagenteComercial

**Autores:** Owen Badel, Maverick Diaz, Jesus Viloria  
**Docente:** Juan Puello  
**Institucion:** Fundacion Internacional Colombo (Unicolombo)  
**Ecosistema tecnologico:** Python 3.12, SPADE (`spade`), Scikit-Learn (`scikit-learn`), FastAPI, Uvicorn y Meta Graph API (WhatsApp Cloud API, Instagram Direct y Facebook Messenger).

---

## 1. Descripcion General

`SistemaMultiagenteComercial` implementa una arquitectura multiagente completa basada en el modelo organizacional **MAS-ML** y el estandar de comunicacion **FIPA-ACL** sobre SPADE (con servidor XMPP embebido PyJabber), conectada en tiempo real con **Meta Platforms** mediante un servidor FastAPI Webhook y un panel web de supervision **Human-in-the-Loop**.

El sistema sustituye cualquier simulacion por consola y recibe mensajes reales provenientes de **WhatsApp**, **Instagram** y **Messenger** a traves de `POST /webhook`, ejecuta el razonamiento colaborativo entre los 5 agentes inteligentes y responde automaticamente al canal de Meta correspondiente utilizando `httpx` contra la Graph API oficial.

---

## 2. Arquitectura Organizacional y Entorno (MAS-ML)

- **Organizacion (`<<organization>> SistemaMultiagenteComercial`):** Define por composicion (`<<owns>>`) los 5 roles del sistema y aplica las reglas de escalamiento obligatorio hacia `Coordinador` y el canal centralizado de promociones.
- **Entorno (`<<environment>> EntornoVirtualAtencion`):** Espacio compartido donde habitan (`<<inhabit>>`) la organizacion y los 5 agentes:
  - `plataformas`: conectores de redes sociales (`WhatsApp`, `Instagram`, `Facebook`) enlazados con `ClienteMetaGraphAPI`.
  - `perfilesClientes`: memoria conversacional, tono, historial, cotizacion vigente, parametros parciales y bandera `automatizacion_activa` para control Human-in-the-Loop.
  - `tarifasYMargenes`: historico de tarifas (`45` registros) e historico de ofertas (`45` registros) con topes de descuento y margenes de ganancia.
  - `agendaVirtual`: disponibilidad de bloques horarios y reservas confirmadas.
  - Eventos del entorno: `eventoLlegadaMensaje()` y `eventoCambioDisponibilidad()`.

### Agentes, Roles, Arquitecturas y Modelos ML

| Agente (`<<agent>>`) | Rol (`<<role>>`) | Arquitectura | Motor de Machine Learning | Relacion `<control>` |
| :--- | :--- | :--- | :--- | :--- |
| `AgenteCoordinador` | `Coordinador` | Basado en utilidad | Reglas de utilidad y control de margenes | Supervisa a los otros 4 roles |
| `AgenteVentas` | `EjecutivoVentas` | Con aprendizaje + Utilidad | NLP (`TfidfVectorizer` + `LogisticRegression`) | Subordinado a `Coordinador` |
| `AgenteCotizaciones` | `LiquidadorCotizaciones` | Con aprendizaje + Metas | `RandomForestRegressor` (Scikit-Learn) | Subordinado a `Coordinador` |
| `AgentePromociones` | `AnalistaPromociones` | Con aprendizaje + Utilidad | `KNeighborsRegressor` (KNN Scikit-Learn) | Subordinado a `Coordinador` |
| `AgenteAgendamientos` | `GestorAgenda` | Basado en metas | Consulta y validacion de agenda | Subordinado a `Coordinador` |

---

## 3. Diagramas de Secuencia y Estandar FIPA-ACL

Todos los mensajes entre agentes utilizan `spade.message.Message` con:
- `ontology = "Comercial-SMA"`
- `protocol` en `{"FIPA-Request", "FIPA-Propose", "FIPA-Query"}`
- Registro dual en consola enriquecida y en el Monitor Web en tiempo real (`GET /`): formato estructurado JSON/ACL y etiqueta de 3 lineas exigida por FIPA-ACL:

```text
[AgenteEmisor -> AgenteReceptor]
PERFORMATIVA
(predicado_contenido parametros)
<<ACL:PERFORMATIVA>>
```

Flujos operativos soportados de extremo a extremo desde Meta:
1. **Escenario 1 (`FIPA-Request`):** Atencion inicial con catalogo visual, extraccion de parametros de servicio, escalamiento a `AgenteCoordinador`, calculo con `RandomForestRegressor` en `AgenteCotizaciones`:
   - `alt [Calculo exitoso]`: `AGREE` -> `INFORM (cotizacion_calculada ...)` -> respuesta al cliente en Meta.
   - `else [Faltan parametros obligatorios]`: `FAILURE (motivo: parametros_incompletos)` -> `REQUEST (solicitar_parametros_faltantes cliente)` -> el cliente envia los datos faltantes en el siguiente mensaje y el sistema fusiona los parametros automaticamente.
2. **Escenario 2 (`FIPA-Propose`):** Deteccion de objecion/indecision (`tonoCliente = indeciso`), solicitud al `AgenteCoordinador`, calculo de descuento optimo con `KNeighborsRegressor` en `AgentePromociones`:
   - `alt [acepta - respeta margen]`: `PROPOSE` -> `ACCEPT-PROPOSAL` -> `INFORM` -> `AgenteVentas` presenta el incentivo al cliente en Meta.
   - `else [rechaza - compromete rentabilidad]`: `REJECT-PROPOSAL` -> `REFUSE (motivo: descuento_no_viable)` -> se mantiene la tarifa base.
3. **Agendamiento (`FIPA-Query`):** `QUERY-IF (consultar_disponibilidad ...)` -> `AGREE` / `CONFIRM` (con horarios alternativos cuando el bloque esta ocupado).
4. **Escalamiento Human-in-the-Loop:** Ante cliente `hostil`, solicitud de soporte humano o regla critica, `AgenteCoordinador.pausarYAlertarHumano()` pausa la automatizacion del cliente (`automatizacion_activa = False`), emite una alerta prioritaria al WhatsApp del administrador (`ADMIN_WHATSAPP_NUMBER` si esta configurado) y habilita la toma de control desde el panel web.

---

## 4. Estructura del Proyecto

```text
SistemaMultiagenteComercial/
|-- .env.example
|-- main.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- tarifas_historicas.csv
|   `-- historico_ofertas.csv
|-- src/
|   |-- agentes/
|   |   |-- base_agente.py
|   |   |-- agente_coordinador.py
|   |   |-- agente_ventas.py
|   |   |-- agente_cotizaciones.py
|   |   |-- agente_promociones.py
|   |   `-- agente_agendamientos.py
|   |-- comunicacion/
|   |   |-- fipa_acl.py
|   |   |-- monitor_fipa.py
|   |   `-- spade_runtime.py
|   |-- entorno/
|   |   `-- entorno_virtual.py
|   |-- escenarios/
|   |   `-- orquestador_escenarios.py
|   |-- integracion_meta/
|   |   |-- configuracion.py
|   |   |-- cliente_meta.py
|   |   `-- webhook_server.py
|   |-- ml/
|   |   |-- datasets.py
|   |   `-- modelos_ml.py
|   `-- organizacion/
|       |-- roles.py
|       `-- organizacion_comercial.py
`-- tests/
    `-- test_sistema_multiagente.py
```

---

## 5. Configuracion de Credenciales Meta y Ejecucion

### Instalacion de dependencias

```bash
pip install -r requirements.txt
```

### Configuracion `.env` para Meta Platforms

Copia `.env.example` como `.env` en la raiz del proyecto y configura las credenciales de tu aplicacion en Meta for Developers:

```bash
copy .env.example .env
```

Variables principales en `.env`:
- `META_VERIFY_TOKEN`: Token secreto para validar la suscripcion del webhook (`GET /webhook`).
- `META_APP_SECRET`: Opcional recomendado para validar la firma `X-Hub-Signature-256` en cada `POST /webhook`.
- `META_API_VERSION`: Version de Meta Graph API (ej. `v21.0`).
- `WHATSAPP_ACCESS_TOKEN` y `WHATSAPP_PHONE_NUMBER_ID`: Credenciales de WhatsApp Cloud API.
- `ADMIN_WHATSAPP_NUMBER`: Numero WhatsApp del asesor humano que recibe alertas cuando se dispara `pausarYAlertarHumano()`.
- `MESSENGER_PAGE_ACCESS_TOKEN`: Token de pagina para Facebook Messenger.
- `INSTAGRAM_ACCESS_TOKEN` e `INSTAGRAM_ACCOUNT_ID`: Credenciales para Instagram Direct Messaging.

### Ejecucion del servidor real (SPADE + XMPP Embebido + FastAPI)

```bash
python main.py
```

El comando arranca con un solo paso:
- Servidor XMPP embebido (`localhost:5222`) con los 5 agentes SPADE conectados.
- Webhook oficial en `http://127.0.0.1:8000/webhook`.
- Panel Web Monitor FIPA-ACL y consola Human-in-the-Loop en `http://127.0.0.1:8000/`.

Para exponer el webhook local hacia Meta en desarrollo se recomienda crear un tunel HTTPS (por ejemplo `ngrok http 8000` o Cloudflare Tunnel) y registrar `<URL_HTTPS>/webhook` junto con `META_VERIFY_TOKEN` en el panel de **Meta for Developers** suscribiendose al campo `messages` en WhatsApp, Messenger e Instagram.

Parametros opcionales de arranque:
- `--host 0.0.0.0`
- `--port 8000`
- `--local-bus` (utiliza el bus asincrono local en caso de conflicto en el puerto XMPP `5222`).

---

## 6. Pruebas Automatizadas

Ejecutar la suite completa con `pytest`:

```bash
pytest -v
```
