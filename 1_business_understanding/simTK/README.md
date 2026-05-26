# Entendimiento del problema — SimTK `cp-child-gait`

## ¿Qué es la parálisis cerebral?

La parálisis cerebral (PC) es un grupo de trastornos del movimiento y la postura causados por una lesión no progresiva en el cerebro en desarrollo (prenatal, perinatal o postnatal temprana). Es la causa más común de discapacidad motora en la infancia. Las alteraciones en el control motor se manifiestan principalmente en la marcha, cuyo patrón difiere de forma característica entre subtipos clínicos.

---

## Grupos clínicos del dataset

El dataset contiene tres grupos. Ver diagrama: [`diagrams/grupos_clinicos.svg`](figs/simtk_cp_child_gait_dataset_diagram.svg)

### Hemiplejia espástica — n=5, edad 9.0 ± 2.3 años

Compromiso unilateral: un lado del cuerpo (brazo y pierna del mismo lado). La marcha presenta:

- Asimetría bilateral marcada
- Circunducción de cadera en el lado afectado
- Pie equino (flexión plantar)
- Reducción del rango de flexión de rodilla en la fase de balanceo
- Lado contralateral con marcha normal o casi normal

### Diplejia espástica — n=4, edad 10.5 ± 1.7 años

Compromiso bilateral con predominio en miembros inferiores. Es el subtipo más prevalente de PC espástica. Patrones de marcha comunes:

- *Crouch gait*: flexión excesiva de cadera y rodilla durante todo el ciclo
- *Stiff-knee gait*: reducción del pico de flexión de rodilla en balanceo
- *Jump gait*: flexión de cadera y rodilla con plantar flexión simultánea
- Marcha en tijera por aducción de cadera, base de sustentación estrecha

### Desarrollo típico (control) — n=5, edad 8.4 ± 1.5 años

Niños sin patología neurológica ni musculoesquelética. Sirven como referencia de normalidad.

**Criterio de inclusión compartido:** sin cirugía ni toxina botulínica en los 6 meses previos.

---

## Clasificación funcional: GMFCS

El Gross Motor Function Classification System (GMFCS) estratifica la severidad en 5 niveles. Ver diagrama: [`diagrams/gmfcs.svg`](diagrams/gmfcs.svg)

| Nivel | Descripción |
|-------|-------------|
| I | Camina sin restricciones; limitaciones en habilidades avanzadas |
| II | Camina con limitaciones en terrenos irregulares o largas distancias |
| III | Camina con dispositivo de ayuda manual |
| IV | Movilidad autónoma limitada; usa silla de ruedas en exteriores |
| V | Transporte manual en todos los contextos |

Los sujetos de `cp-child-gait` corresponden a **niveles I–II**. El proyecto del exoesqueleto apunta a **niveles II–IV**.

---

## Señales disponibles

| Señal | Instrumento | Uso |
|-------|-------------|-----|
| Cinemática 3D (ángulos articulares) | VICON — marcadores reflectivos | Features temporales |
| Fuerzas de reacción del suelo (GRF) | Plataformas de fuerza AMTI | Derivar eventos de marcha (FC/FO) |
| Datos estáticos | Medición antropométrica | Normalización |

Formato: `.c3d` (legible con `ezc3d` en Python).

---

## Objetivos de modelado

**Tarea 1 — Detección de fase de marcha**
Segmentar automáticamente el ciclo en sus fases principales usando ángulos articulares y GRF.

**Tarea 2 — Clasificación de tipo de PC**
Distinguir entre hemiplejia espástica, diplejia espástica y desarrollo típico a partir de features extraídas por ciclo.

**Objetivo a largo plazo**
Que los modelos entrenados sobre datos VICON/AMTI sean replicables con encoders y FSR del exoesqueleto pediátrico, minimizando el domain shift.

---

## Referencias

- SimTK `cp-child-gait`: https://simtk.org/projects/cp-child-gait
- GMFCS: Palisano et al. (1997, revisado 2007)
