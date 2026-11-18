# SistemaMultiagenteComercial

**Autores:** Owen Badel, Maverick Diaz, Jesus Viloria  
**Docente:** Juan Puello  
**Institucion:** Fundacion Internacional Colombo (Unicolombo)  
**Ecosistema tecnologico:** Python 3.12, SPADE (`spade`), Scikit-Learn (`scikit-learn`), FastAPI, Uvicorn y Botpress (WhatsApp, Webchat, Facebook Messenger e Instagram).

---

## 1. Descripcion General

`SistemaMultiagenteComercial` implementa una arquitectura multiagente basada en el modelo organizacional **MAS-ML** y el estandar de comunicacion **FIPA-ACL** sobre SPADE (con servidor XMPP embebido PyJabber), conectada como backend inteligente de **Botpress**.

Botpress actua unicamente como pasarela de canales (WhatsApp, Webchat, Messenger e Instagram) enviando cada mensaje entrante a `POST /api/botpress/mensaje`. Todo el procesamiento NLP, la comunicacion FIPA-ACL entre los 5 agentes, las predicciones con Scikit-Learn y el control Human-in-the-Loop ocurren dentro del sistema multiagente en Python.

---

## 2. Arquitectura Organizacional y Entorno (MAS-ML)

- **Organizacion (`<<organization>> SistemaMultiagenteComercial`):** Define por composicion (`<<owns>>`) los 5 roles del sistema y aplica las reglas de escalamiento obligatorio hacia `Coordinador` y el canal centralizado de promociones.
- **Entorno (`<<environment>> EntornoVirtualAtencion`):** Espacio compartido donde habitan (`<<inhabit>>`) la organizacion y los 5 agentes:
  - `plataformas`: registro multicanal (`WhatsApp`, `Webchat`, `Facebook`, `Instagram`).
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

Flujos operativos soportados de extremo a extremo desde Botpress:
1. **Escenario 1 (`FIPA-Request`):** Atencion inicial con catalogo visual, extraccion de parametros de servicio, escalamiento a `AgenteCoordinador`, calculo con `RandomForestRegressor` en `AgenteCotizaciones`:
   - `alt [Calculo exitoso]`: `AGREE` -> `INFORM (cotizacion_calculada ...)` -> respuesta al cliente.
   - `else [Faltan parametros obligatorios]`: `FAILURE (motivo: parametros_incompletos)` -> `REQUEST (solicitar_parametros_faltantes cliente)` -> el cliente envia los datos faltantes en el siguiente turno y el sistema fusiona los parametros automaticamente.
2. **Escenario 2 (`FIPA-Propose`):** Deteccion de objecion/indecision (`tonoCliente = indeciso`), solicitud al `AgenteCoordinador`, calculo de descuento optimo con `KNeighborsRegressor` en `AgentePromociones`:
   - `alt [acepta - respeta margen]`: `PROPOSE` -> `ACCEPT-PROPOSAL` -> `INFORM` -> `AgenteVentas` presenta el incentivo al cliente.
   - `else [rechaza - compromete rentabilidad]`: `REJECT-PROPOSAL` -> `REFUSE (motivo: descuento_no_viable)` -> se mantiene la tarifa base.
3. **Agendamiento (`FIPA-Query`):** `QUERY-IF (consultar_disponibilidad ...)` -> `INFORM` -> `REQUEST` -> `AGREE` -> `CONFIRM` (o `PROPOSE` con horarios alternativos si el bloque esta ocupado).
4. **Escalamiento Human-in-the-Loop:** Ante cliente `hostil` o reclamo critico, `AgenteCoordinador.pausarYAlertarHumano()` pausa la automatizacion del cliente (`automatizacion_activa = False`), devuelve `"pausado_humano": true` hacia Botpress y habilita la toma de control desde el panel web (`/`).

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
|   |-- integracion_botpress/
|   |   |-- configuracion.py
|   |   `-- servidor_botpress.py
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

## 5. Ejecucion y Conexion con Botpress

### Instalacion y arranque

```bash
pip install -r requirements.txt
python main.py
```

Esto inicia:
- Servidor XMPP embebido (`localhost:5222`) con los 5 agentes SPADE conectados.
- API puente para Botpress en `http://127.0.0.1:8000/api/botpress/mensaje`.
- Panel Web Monitor FIPA-ACL y consola Human-in-the-Loop en `http://127.0.0.1:8000/`.

Para conectar Botpress Cloud con tu PC local, expone el puerto `8000` por HTTPS:

```bash
ngrok http 8000
```

En tu flujo de **Botpress Studio**, agrega una tarjeta **Execute Code** conectada a la respuesta del usuario que haga `POST` hacia `https://<tu-url-ngrok>/api/botpress/mensaje` enviando `{ canal, id_cliente, nombre, texto }` y devuelva `response.data.respuesta` al usuario.

---

## 6. Pruebas Automatizadas

```bash
pytest -v
```
