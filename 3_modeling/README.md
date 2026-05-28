# Modelado

Este directorio contiene las propuestas y experimentos de modelado para el análisis de marcha en niños con parálisis cerebral.

El objetivo central es aprender representaciones latentes de la marcha a partir de datos VICON, usando transferencia de conocimiento desde modelos preentrenados en grandes cohortes clínicas hacia los datasets disponibles en este proyecto.

---

## Estructura

```
modeling/
├── README.md                        # Este archivo — visión general
└── 01_intellevent_to_ferrari/       # Propuesta completa de transfer learning
    └── README.md
```

---

## Aproximaciones a explorar

- **Transferencia desde IntellEvent** — uso de los pesos BiLSTM preentrenados en detección de eventos IC/FO como punto de partida para clasificación de patrones de marcha
- **Fine-tuning supervisado** — adaptación al problema de clasificación de 4 formas Ferrari (diplejia espástica) con validación LOSO
- **Análisis de embeddings** — exploración del espacio latente aprendido sobre datos independientes (SimTK), incluyendo visualización t-SNE/UMAP y separación entre grupos clínicos
- **Baselines de referencia** — clasificación sobre features clásicas (ángulos articulares normalizados a 100 puntos del ciclo) para cuantificar el aporte de las representaciones temporales crudas

La descripción detallada de cada aproximación, incluyendo arquitectura, preprocesamiento, estrategia de entrenamiento y métricas de evaluación, se documenta en el README del subfolder correspondiente.
