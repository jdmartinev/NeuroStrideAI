# Entendimiento de los datos — SimTK `cp-child-gait`

## Fuente

**SimTK `cp-child-gait`** — https://simtk.org/projects/cp-child-gait  
Formato de archivo: `.c3d` (biomecánica estándar). Librería de lectura en Python: `ezc3d`.

---

## Sensores e instrumentación

### Sistema de captura de movimiento — VICON

Sistema óptico de marcadores reflectivos de alta precisión. Los marcadores se adhieren a puntos anatómicos específicos del cuerpo y son rastreados por múltiples cámaras infrarrojas sincronizadas.

**Modelo de marcadores:** Plug-in Gait (full body) — estándar clínico para análisis de marcha.

**Qué mide:** posición 3D de cada marcador en el espacio (x, y, z) a lo largo del tiempo. A partir de estas posiciones se calculan los ángulos articulares de cadera, rodilla y tobillo en los tres planos de movimiento:

| Plano | Movimiento |
|-------|------------|
| Sagital | Flexión / extensión |
| Frontal | Abducción / aducción |
| Transversal | Rotación interna / externa |

**Frecuencia de muestreo:** 100 Hz (típica para sistemas VICON en análisis de marcha pediátrica).

---

### Plataformas de fuerza — AMTI

Plataformas embebidas en el suelo que miden las fuerzas que el pie ejerce contra la superficie durante la marcha.

**Qué miden:** Fuerzas de reacción del suelo (GRF — *Ground Reaction Forces*) en tres componentes:

| Componente | Dirección | Descripción |
|------------|-----------|-------------|
| Fx | Mediolateral | Fuerza lateral |
| Fy | Anteroposterior | Fuerza de frenado / propulsión |
| Fz | Vertical | Soporte de peso corporal |

El componente vertical (Fz) es el más relevante para derivar los eventos del ciclo de marcha: **foot contact (FC)** y **foot off (FO)**, que definen el inicio y fin de las fases de apoyo y balanceo.

**Frecuencia de muestreo:** 1000 Hz (típica para plataformas de fuerza), submuestreada o sincronizada con VICON en el archivo `.c3d`.

---

### Datos estáticos

Además de los trials de marcha dinámica, el dataset incluye una captura estática por sujeto. Se usa para:

- Calibración del modelo antropométrico
- Cálculo de longitudes segmentarias (fémur, tibia, pie)
- Normalización de ángulos articulares respecto a la postura de referencia del sujeto

---

## Resumen de señales disponibles

| Señal | Sensor | Unidad | Uso principal |
|-------|--------|--------|---------------|
| Posición 3D de marcadores | VICON | mm | Cálculo de ángulos articulares |
| Ángulos articulares (cadera, rodilla, tobillo) | Derivado de VICON | grados | Features para modelos |
| GRF — Fx, Fy, Fz | AMTI | N (o N/kg normalizado) | Derivar eventos FC/FO |
| Datos estáticos | VICON + AMTI | — | Normalización por sujeto |

!(Summary data)[figs/simTK_data.png]

---

## Siguientes pasos

Con la instrumentación clara, el siguiente paso es el inventario real de archivos: cuántos trials por sujeto, qué señales están efectivamente presentes en cada `.c3d`, y una primera revisión de calidad.

→ Ver `eda/01_inventory.py`
