# SistemaMultiagenteComercial (MAS-ML + SPADE + FIPA-ACL + Scikit-Learn)

> **Autoría Principal:** Owen Badel Hooker — Ingeniero de Sistemas  
> **Equipo Académico:** Owen Badel Hooker, Maverick Díaz, Jesús Viloria  
> **Asignatura:** Inteligencia Artificial | **Docente:** Juan Puello | **Institución:** Fundación Internacional Colombo Unicolombo

Implementación funcional en Python de un **Sistema Multiagente Cooperativo (SMA)** para la automatización conversacional, cálculo inteligente de cotizaciones, intermediación de promociones cuidando la rentabilidad y agendamiento de citas para profesionales independientes. El proyecto traduce de manera fiel el **Diagrama de Clases MAS-ML**, las tablas de análisis conceptual y los **Diagramas de Secuencia FIPA-ACL** sobre el framework **SPADE** (con servidor XMPP embebido PyJabber) y modelos predictivos de **Scikit-Learn**.

---

## 🏛️ Arquitectura MAS-ML Implementada

1. **Entorno Compartido (`<<environment>> EntornoVirtualAtencion`):**
   - `plataformas : RedesSociales` (WhatsApp, Instagram, Facebook + portafolio visual).
   - `perfilesClientes : Informacion` (estado conversacional, tono, fidelidad e historial).
   - `tarifasYMargenes : BaseDatos` (histórico de 45 cotizaciones, 45 ofertas y márgenes mínimos de ganancia del 22% con tope de descuento del 15%).
   - `agendaVirtual : Calendario` (gestión de bloques libres y ocupados).
   - Eventos propios del entorno: `eventoLlegadaMensaje()` y `eventoCambioDisponibilidad()`.

2. **Organización (`<<organization>> SistemaMultiagenteComercial`):**
   - Posee por composición (`<<owns>>`) los 5 roles: `Coordinador`, `EjecutivoVentas`, `LiquidadorCotizaciones`, `AnalistaPromociones` y `GestorAgenda`.
   - Habita (`<<inhabit>>`) el `EntornoVirtualAtencion`.
   - Impone las reglas de **escalamiento obligatorio al Coordinador**, **canal centralizado de promociones** y las **5 dependencias formales**:
     - `Coordinador <--- LiquidadorCotizaciones`
     - `Coordinador <--- AnalistaPromociones`
     - `EjecutivoVentas <--- Coordinador`
     - `GestorAgenda <--- LiquidadorCotizaciones`
     - `Coordinador <--- GestorAgenda`

3. **Los 5 Agentes Inteligentes SPADE y Modelos ML:**
   - **`AgenteCoordinador` (`Coordinador`)**: Arquitectura basada en utilidad; supervisa (`<<control>>`) a los 4 especialistas, evalúa rentabilidad de promociones y ejecuta escalamiento humano.
   - **`AgenteVentas` (`EjecutivoVentas`)**: Arquitectura con aprendizaje NLP (`TfidfVectorizer` + `LogisticRegression` para clasificar intención y `tonoCliente`: `interesado`, `indeciso`, `hostil`) y basado en utilidad.
   - **`AgenteCotizaciones` (`LiquidadorCotizaciones`)**: Arquitectura basada en metas con **`RandomForestRegressor`** entrenado sobre `data/tarifas_historicas.csv` (45 registros).
   - **`AgentePromociones` (`AnalistaPromociones`)**: Arquitectura basada en utilidad con **`KNeighborsRegressor` (Regresión KNN)** entrenado sobre `data/historico_ofertas.csv` (45 registros).
   - **`AgenteAgendamientos` (`GestorAgenda`)**: Arquitectura basada en metas para prevención de cruces horarios y confirmación de citas (`QUERY-IF`, `AGREE`, `CONFIRM`).

---

## 📂 Estructura Modular del Proyecto

```text
SistemaMultiagenteComercial/
├── main.py                              # Punto de entrada con menú interactivo y SPADE XMPP embebido
├── requirements.txt                     # Dependencias Python (spade, scikit-learn, pandas, rich, pytest)
├── data/
│   ├── tarifas_historicas.csv           # Dataset sintético de cotizaciones (45 registros)
│   └── historico_ofertas.csv            # Dataset sintético de ofertas y descuentos (45 registros)
├── src/
│   ├── entorno/entorno_virtual.py       # <<environment>> EntornoVirtualAtencion
│   ├── organizacion/
│   │   ├── roles.py                     # Los 5 <<role>> MAS-ML (Deberes, Derechos, Restricciones)
│   │   └── organizacion_comercial.py    # <<organization>> SistemaMultiagenteComercial
│   ├── comunicacion/
│   │   ├── fipa_acl.py                  # Constructor de spade.message.Message FIPA-ACL
│   │   ├── monitor_fipa.py              # Logger / Monitor FIPA-ACL en tiempo real (Rich)
│   │   └── spade_runtime.py             # Runtime SPADE XMPP embebido y contenedor local
│   ├── ml/
│   │   ├── datasets.py                  # Generador de datasets y corpus NLP en español
│   │   └── modelos_ml.py                # NLP (TF-IDF), RandomForestRegressor y KNeighborsRegressor
│   ├── agentes/
│   │   ├── base_agente.py               # Clase base sobre spade.agent.Agent
│   │   ├── agente_coordinador.py        # AgenteCoordinador
│   │   ├── agente_ventas.py             # AgenteVentas
│   │   ├── agente_cotizaciones.py       # AgenteCotizaciones
│   │   ├── agente_promociones.py        # AgentePromociones
│   │   └── agente_agendamientos.py      # AgenteAgendamientos
│   └── escenarios/
│       └── orquestador_escenarios.py    # Orquestador E2E de todos los Diagramas de Secuencia
└── tests/
    └── test_sistema_multiagente.py      # Suite de pruebas unitarias y E2E
```

---

## 🚀 Instalación y Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el sistema con el servidor XMPP embebido de SPADE (sin servidores externos)
python main.py

# 3. Opciones rápidas por línea de comandos
python main.py --demo-all      # Ejecuta automáticamente los casos 1A, 1B, 2A, 2B, 3A, 3B y 4
python main.py --caso 2A       # Ejecuta un escenario específico directamente

# 4. Ejecutar la suite de pruebas automatizadas
pytest -v
```
