# Entendimiento de los datos — IntellEvent (pesos preentrenados)

## Fuente

**IntellEvent** — Dumphart et al., *PLOS ONE* 2023  
Repositorio: `github.com/fhstp/IntellEvent`  
DOI: `10.1371/journal.pone.0288555`  
Institución: St. Pölten University of Applied Sciences + Orthopaedic Hospital Vienna-Speising, Austria

> Los datos originales **no están disponibles públicamente** (política del hospital). Lo que se publica son los pesos del modelo entrenado y el código de inferencia.

---

## Datos de entrenamiento (no accesibles, documentados en el paper)

### Población

| Propiedad | Valor |
|---|---|
| N total | 1211 pacientes + 61 controles sanos |
| Institución | Orthopaedic Hospital Vienna-Speising |
| Patologías | Malrotación de miembros inferiores, pie zambo, parálisis cerebral infantil (PCI), PCI con drop foot |
| Tipo de marcha | Marcha overground a velocidad preferida |
| Captura | VICON (frecuencia no especificada en el paper, típicamente 100–200 Hz) |

---

## Sensores e instrumentación

### Sistema de captura — VICON

Sistema óptico con marcadores reflectivos. IntellEvent usa únicamente **3 marcadores** de los disponibles:

| Marcador | Ubicación anatómica |
|---|---|
| HEEL | Calcáneo (talón) |
| TOE | 5.º metatarsiano (punta del pie) |
| ANKLE | Maléolo lateral (tobillo) |

Esto equivale, en la nomenclatura de Lucabergamini, a los marcadores `RCA/LCA`, `RFM/LFM`, y una aproximación de `RLE/LLE`.

**Qué se calcula a partir de estos marcadores:**
- Posición 3D: `(x, y, z)` por marcador
- Velocidad: derivada por diferencias finitas `(vx, vy, vz)` por marcador

### Construcción del vector de entrada

```
3 marcadores × 6 canales (x, y, z, vx, vy, vz) = 36 canales por frame
```

**Forma del tensor de entrada:** `(T, 36)` donde `T` es la longitud del trial en frames, rellenado con padding hasta `T_max = 577` (el trial más largo del conjunto de entrenamiento).

> **Importante:** Los datos de entrenamiento son secuencias temporales **crudas, sin normalización de ciclo de marcha**. No se aplica remuestreo a 100 puntos ni DTW. Esto es una consecuencia necesaria: la detección de eventos de marcha es un prerequisito para la normalización de ciclo, por lo que normalizar antes sería circular.

---

## Arquitectura del modelo

Dos modelos BiLSTM independientes, uno por tipo de evento:

| Componente | Detalle |
|---|---|
| Arquitectura | Bidirectional LSTM (BiLSTM) |
| Modelos | 2 — uno para IC, uno para FO |
| Entrada | `(T, 36)` — posición + velocidad de 3 marcadores |
| Salida | Secuencia de probabilidades por frame: `(T, 3)` |
| Clases de salida | 0 = sin evento / 1 = evento izquierdo / 2 = evento derecho |
| Post-procesamiento | Detección de picos con umbral = 0.01 |

---

## Etiquetas de entrenamiento

Supervisión temporal densa — una etiqueta por frame:

| Evento | Descripción clínica | Error medio | Tasa de detección |
|---|---|---|---|
| IC — Initial Contact | Contacto inicial del talón con el suelo | ≤ 5.4 ms | ≥ 99% |
| FO — Foot Off | Despegue de los dedos del suelo | ≤ 11.3 ms | ≥ 95% |

Las etiquetas originales fueron asignadas manualmente por clínicos expertos o derivadas de plataformas de fuerza (cuando estaban disponibles).

---

## Archivos del repositorio

```
IntellEvent/
├── model_IC.pt          # Pesos BiLSTM para Initial Contact
├── model_FO.pt          # Pesos BiLSTM para Foot Off
├── predict.py           # Inferencia sobre nuevos trials
├── train.py             # Reentrenamiento / fine-tuning
└── nexus_pipeline/      # Integración con Vicon Nexus 2.14+
```

---

## Uso en este pipeline

IntellEvent se usa en **dos roles distintos**:

### Rol 1 — Extractor de eventos (Stage 1, inferencia directa)

Aplicar los pesos tal como están sobre los trials de Lucabergamini y SimTK para obtener eventos IC y FO. Esto permite segmentar cada trial en ciclos individuales sin necesidad de plataformas de fuerza.

**Restricción:** IntellEvent fue entrenado con 3 marcadores específicos. Para aplicarlo a Lucabergamini (19 marcadores) o SimTK (39 marcadores), se deben extraer los marcadores equivalentes y calcular sus velocidades con el mismo esquema.

### Rol 2 — Encoder preentrenado (Stage 2, transferencia de pesos)

El BiLSTM de IntellEvent se usa como punto de partida para el fine-tuning sobre Lucabergamini. Se añade una **capa de proyección de entrada** (`Linear(114 → 36)`) para adaptar los 19 marcadores al espacio de entrada esperado por el BiLSTM, y una **cabeza de clasificación** para las 4 clases de Ferrari.

La capa de proyección y la cabeza se entrenan desde cero. El BiLSTM se congela en la fase de calentamiento y se descongela con learning rate muy bajo en la fase de fine-tuning.

**Lo que aporta la transferencia:** El encoder ya aprendió a interpretar trayectorias de marcadores de pie/tobillo en pacientes con patologías de marcha. No parte de cero frente a señales biomecánicas.

---

## Consideraciones de dominio

| Aspecto | Valor |
|---|---|
| País / laboratorio | Austria (Vienna-Speising) |
| Patologías incluidas en entrenamiento | PCI (infantil), malrotación, pie zambo, drop foot |
| Patologías **no** incluidas | Diplegia clasificada por formas Ferrari, hemiplegia |
| Frecuencia de captura | No publicada (los autores advierten que modelos entrenados en un laboratorio deben aplicarse con cuidado en otro por diferencias de setup y frecuencia) |

---

## Referencias

- Dumphart B, Slijepcevic D, Zeppelzauer M, Kranzl A, Unglaube F, Baca A, Horsak B. (2023). Robust deep learning-based gait event detection across various pathologies. *PLOS ONE*, 18(8), e0288555. `doi.org/10.1371/journal.pone.0288555`
