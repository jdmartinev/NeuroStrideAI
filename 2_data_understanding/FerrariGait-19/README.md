# Entendimiento de los datos — Lucabergamini (`gait-analysis-dataset`)

## Fuente

**Lucabergamini gait-analysis-dataset** — Ferrari, Bergamini et al., 2019  
Repositorio: `github.com/lucabergamini/gait-analysis-dataset`  
Papers asociados:
- Ferrari A, Bergamini L, et al. (2019). Gait-based diplegia classification using LSTM networks. *Journal of Healthcare Engineering*, 2019, 3796898. `doi.org/10.1155/2019/3796898`
- Bergamini L, et al. (2017). Signal processing and machine learning for diplegia classification. *ICIAP 2017*, LNCS 10485, 97–108.

---

## Población

| Propiedad | Valor |
|---|---|
| N pacientes | 178 |
| N trials | 1139 |
| Diagnóstico | Diplejia espástica (exclusivamente) |
| Institución | Hospital italiano (no especificado públicamente) |
| Acceso | Abierto — descarga directa desde el repositorio |

> El dataset **no incluye** controles sanos ni pacientes con hemiplejia. Todos los sujetos presentan diplejia espástica clasificable en las 4 formas Ferrari.

---

## Sensores e instrumentación

### Sistema de captura — VICON

Sistema óptico de alta frecuencia con marcadores reflectivos. A diferencia del Plug-in-Gait completo (39 marcadores), este dataset usa un **conjunto reducido de 19 marcadores**.

**Qué mide:** posición 3D de cada marcador `(x, y, z)` a lo largo del tiempo, más un flag de validez por frame.

**Frecuencia de muestreo:** alta frecuencia (el paper menciona "high frequency VICON cameras"; valor exacto no publicado).

**Sin plataformas de fuerza:** el dataset no incluye GRF. Los eventos de marcha no están anotados.

---

## Conjunto de marcadores (19 marcadores)

| Marcador | Ubicación anatómica | Región |
|---|---|---|
| C7 | 7.ª vértebra cervical | Tronco |
| LA, RA | Acromion izquierdo / derecho | Hombros |
| REP, LEP | Epicóndilo lateral derecho / izquierdo | Codos |
| RUL, LUL | Cúbito derecho / izquierdo (muñeca) | Muñecas |
| RASIS, LASIS | EIAS derecha / izquierda | Pelvis anterior |
| RPSIS, LPSIS | EIPS derecha / izquierda | Pelvis posterior |
| RGT, LGT | Trocánter mayor derecho / izquierdo | Cadera |
| RLE, LLE | Epicóndilo lateral de rodilla der. / izq. | Rodillas |
| RCA, LCA | Calcáneo derecho / izquierdo | Talones |
| RFM, LFM | 5.º metatarsiano derecho / izquierdo | Pies |

Con estos 19 marcadores es posible calcular aproximadamente los ángulos de pelvis, cadera, rodilla y tobillo en el plano sagital, además de la orientación general del tronco. No hay marcadores de tibia ni de progresión del pie.

---

## Formato y estructura de los archivos

### Formato: `.npy`

Cada trial es un archivo `.npy` independiente con forma:

```
(T, 19, 4)
```

Donde:
- `T` — número de frames (variable entre trials)
- `19` — número de marcadores
- `4` — canales por marcador: `[x, y, z, valid]`

El cuarto canal (`valid`) es un flag binario: `1` = marcador visible y confiable, `0` = marcador ocluido o reconstruido. Los frames con marcadores inválidos deben tratarse explícitamente en el preprocesamiento.

### Organización del repositorio

```
gait-analysis-dataset/
├── README.md
├── c3d.py              # Lector de archivos .c3d originales
├── c3d2np.py           # Script de conversión .c3d → .npy (define los 19 marcadores)
├── visualizer.py       # Visualizador 3D con Open3D
└── img/
    ├── class0_example.gif
    ├── class1_example.gif
    ├── class2_example.gif
    └── class3_example.gif
```

Los archivos `.npy` se descargan por separado desde el enlace indicado en el README.

---

## Etiquetas — Clasificación Ferrari de diplejia (4 clases)

Sistema de clasificación definido por Ferrari et al. (2005) basado en el patrón de marcha observable en el plano sagital. Validado clínicamente en una cohorte de 467 sujetos.

| Clase (índice) | Nombre clínico | Descripción biomecánica | Dificultad para el clasificador |
|---|---|---|---|
| 0 — Forma I | Equino verdadero | Tobillo en flexión plantar durante todo el apoyo, caderas y rodillas extendidas. Posible recurvatum de rodilla. | Fácil — patrón muy distintivo |
| 1 — Forma II | Marcha en salto (*jump gait*) | Equino de tobillo + flexión de caderas y rodillas. Flexión de rodilla en fase media de apoyo (signo sutil). La más frecuente (~35%). | Difícil — signo principal poco evidente |
| 2 — Forma III | Equino aparente | ROM de tobillo normal, pero flexión excesiva de rodilla y cadera durante el apoyo. Parece caminar de puntillas pero la causa es proximal. | Moderada |
| 3 — Forma IV | Marcha en cuclillas (*crouch gait*) | Dorsiflexión excesiva de tobillo + flexión de rodilla y cadera + tijeras. Forma más severa. | Fácil — patrón muy severo y distintivo |

**Distribución de clases:** La Forma II representa aproximadamente el 35% de los pacientes. Las demás formas tienen menor representación. Se recomienda usar pérdida con pesos de clase o sobremuestreo para compensar el desbalance.

**Solapamiento clínico:** Las Formas II y III son las más difíciles de separar — ambas presentan flexión de rodilla durante el apoyo, con diferente origen (distal vs proximal).

---

## Preprocesamiento para este pipeline

Para usar este dataset como entrada al encoder BiLSTM heredado de IntellEvent, se requiere transformar el formato almacenado `(T, 19, 4)` al formato de entrada del modelo `(T, 114)`:

### Paso 1 — Extraer posición y calcular velocidad

```
xyz   = data[:, :, :3]          # (T, 19, 3) — posición
valid = data[:, :, 3:]          # (T, 19, 1) — flag de validez
vel   = diff(xyz, axis=0)       # (T, 19, 3) — velocidad por diferencias finitas
```

### Paso 2 — Enmascarar frames inválidos

Multiplicar posición y velocidad por el flag de validez para que los marcadores ocluidos contribuyan con cero en lugar de valores espurios.

### Paso 3 — Concatenar y aplanar

```
features = concat([xyz, vel], axis=-1)   # (T, 19, 6)
features = reshape(features, (T, 114))   # (T, 114)
```

### Paso 4 — Padding y masking

Los trials tienen longitud variable. Se usa padding hasta el trial más largo del batch, con una máscara de padding para que el BiLSTM ignore los frames rellenos.

---

## Advertencias de calidad

| Problema | Descripción |
|---|---|
| Trials inválidos | Algunos trials tienen marcadores inválidos durante toda la secuencia. Deben filtrarse antes del entrenamiento. |
| Longitud variable | Cada trial tiene un número diferente de frames. Requiere padding + masking. |
| Sin anotaciones de eventos | No hay IC ni FO anotados. Los ciclos de marcha no están segmentados. Para análisis por ciclo, los eventos deben inferirse con IntellEvent. |
| Sin GRF | No hay plataformas de fuerza. No es posible validar eventos de marcha con señal de fuerza. |
| Sin controles sanos | Todos los sujetos son dipléjicos. No es posible comparar con marcha normal dentro de este dataset. |

---

## Consideraciones de dominio

| Aspecto | Valor |
|---|---|
| País / laboratorio | Italia (hospital no identificado) |
| Patologías | Diplejia espástica únicamente |
| Formas Ferrari incluidas | I, II, III, IV |
| Frecuencia de captura | Alta frecuencia (valor exacto no publicado) |
| Diferencias respecto a IntellEvent | País, hospital, marcadores (3 vs 19), mix de patologías |
| Diferencias respecto a SimTK | País, hospital, marcadores (19 vs 39), ausencia de GRF, ausencia de hemiplejia y controles |

---

## Uso en este pipeline

Este dataset es el **dominio fuente del fine-tuning** (Stage 2). Se usa para:

1. Adaptar el encoder BiLSTM de IntellEvent a 19 marcadores mediante una capa de proyección de entrada
2. Entrenar la cabeza de clasificación de 4 clases (formas Ferrari)
3. Aprender una representación latente de la marcha patológica en niños con diplejia espástica
4. Evaluar con validación cruzada Leave-One-Subject-Out (LOSO)

El encoder resultante es el artefacto central del pipeline: un modelo capaz de proyectar un trial de marcha en un vector de embeddings que codifica el patrón de marcha del paciente.

---

## Referencias

- Ferrari A, Bergamini L, Guerzoni G, Calderara S, Bicocchi N, Vitetta G, Borghi C, Neviani R, Ferrari A. (2019). Gait-based diplegia classification using LSTM networks. *Journal of Healthcare Engineering*, 2019, 3796898. `doi.org/10.1155/2019/3796898`
- Bergamini L, Calderara S, Bicocchi N, Ferrari A, Vitetta G. (2017). Signal processing and machine learning for diplegia classification. In: *International Conference on Image Analysis and Processing*, LNCS 10485, 97–108.
- Ferrari A, et al. (2008). The term diplegia should be enhanced. Part I: a new rehabilitation oriented classification of cerebral palsy. *European Journal of Physical and Rehabilitation Medicine*.
