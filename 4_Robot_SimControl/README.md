# Proyecto G8 – Modelado, simulación y control de marcha

## Exoesqueleto e inteligencia artificial para marcha infantil

Este directorio reúne notebooks reproducibles para el **modelado, la simulación y el control de la interacción humano–exoesqueleto**, desarrollados en el marco del proyecto:

> **Desarrollo de una plataforma tecnológica integrada para la toma de decisiones basadas en datos, enfocada en la detección temprana de patologías de marcha de niños y niñas con parálisis cerebral, que permita fortalecer el ecosistema de salud del Valle de Aburrá con herramientas de inteligencia artificial y terapia robótica asistida por exoesqueletos de rehabilitación de marcha.**

---

## Información institucional

| Campo | Descripción |
|---|---|
| **Entidad líder** | Universidad de Antioquia |
| **Entidades aliadas** | Universidad EAFIT, Institución Universitaria ITM, Universidad CES y Comité de Rehabilitación de Antioquia |
| **Financiación** | Cuarta Convocatoria Conjunta del G8+ en alianza con Ruta N |
| **Épica** | Hardware y exoesqueleto |
| **Actividad** | Realizar el modelado y la simulación humano–robot |
| **Entregable asociado** | Repositorio de Simulación: Modelos en MATLAB/Simulink y Python |
| **Código documental de referencia** | IT-002 – Modelado y simulación humano–robot |
| **Responsables técnicos** | Universidad de Antioquia, Universidad EAFIT y Universidad CES |
| **Entorno principal** | Python, MuJoCo, Jupyter Notebook y Google Colab |

---

## Contenido del directorio

### `Tutorial_Exoesqueleto_MuJoCo_Control_Marcha.ipynb`

Notebook orientado a la construcción y evaluación de un modelo simplificado de exoesqueleto de miembro inferior.

Incluye, según la versión disponible:

- modelo planar de cadera y rodilla;
- formulación cinemática y dinámica;
- generación de trayectorias articulares de marcha;
- control proporcional–derivativo;
- control por torque computado;
- simulación de perturbaciones externas;
- análisis de seguimiento y robustez;
- cálculo de métricas de error;
- visualización del movimiento;
- análisis de fuerzas de reacción del suelo, cuando corresponda.

### `Tutorial_x2_human_mujoco_Demo_Marcha.ipynb`

Notebook orientado al uso del modelo `x2_human_mujoco` dentro del entorno MuJoCo.

Incluye, según la versión disponible:

- carga e inspección del modelo MJCF/XML;
- identificación de articulaciones, sensores y actuadores;
- selección robusta de grados de libertad;
- control articular de cadera y rodilla;
- generación de trayectorias periódicas;
- simulación de marcha;
- renderizado de resultados;
- evaluación mediante MAE, RMSE y curvas de seguimiento.

---

## Objetivo técnico

Construir una línea base reproducible de modelado y simulación humano–robot que permita estudiar, antes de la fabricación del prototipo alfa:

1. la arquitectura biomecatrónica;
2. la cinemática y dinámica del sistema;
3. el seguimiento de trayectorias de marcha;
4. las estrategias de control articular;
5. la interacción física humano–exoesqueleto;
6. las restricciones de seguridad;
7. la captura de variables para análisis clínico e inteligencia artificial.

---

## Alcance

Los notebooks se utilizan como **pruebas de concepto computacionales** para:

- evaluar modelos de cadera, rodilla y segmentos corporales;
- comparar estrategias de control;
- estudiar sensibilidad ante perturbaciones;
- documentar parámetros mecánicos y antropométricos;
- producir señales sintéticas de prueba;
- explorar contacto pie–suelo y fuerzas de reacción;
- preparar casos de simulación para control adaptativo e IA explicable;
- apoyar el diseño del prototipo alfa y sus pruebas de banco.

Los desarrollos de esta carpeta no constituyen por sí solos una validación clínica, regulatoria ni de seguridad.

---

## Estructura sugerida del repositorio

```text
.
├── README.md
├── notebooks/
│   ├── Tutorial_Exoesqueleto_MuJoCo_Control_Marcha.ipynb
│   └── Tutorial_x2_human_mujoco_Demo_Marcha.ipynb
├── models/
│   ├── mjcf/
│   └── parameters/
├── data/
│   ├── synthetic/
│   └── examples/
├── figures/
├── videos/
├── results/
├── src/
├── tests/
├── requirements.txt
├── environment.yml
└── LICENSE
```

La estructura puede simplificarse cuando esta carpeta forme parte de un repositorio institucional de mayor tamaño.

---

## Requisitos

Se recomienda utilizar Python 3.10 o superior y un entorno virtual independiente.

Dependencias típicas:

```text
mujoco
numpy
scipy
pandas
matplotlib
jupyter
ipykernel
imageio
imageio-ffmpeg
```

La lista exacta debe conservarse en `requirements.txt` o `environment.yml`, de acuerdo con las versiones verificadas en cada entrega.

---

## Instalación local

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activación en Linux o macOS:

```bash
source .venv/bin/activate
```

Activación en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar Jupyter

```bash
jupyter notebook
```

También puede utilizarse:

```bash
jupyter lab
```

---

## Ejecución en Google Colab

1. Abrir el notebook desde GitHub.
2. Seleccionar **Open in Colab**, cuando el enlace esté disponible.
3. Ejecutar la celda de instalación de dependencias.
4. Verificar la ruta del modelo MuJoCo y de los archivos auxiliares.
5. Ejecutar las celdas en orden.
6. Guardar métricas, figuras y videos en la carpeta de resultados definida.

---

## Resultados esperados

Cada ejecución reproducible debería generar, como mínimo:

- curvas de referencia y respuesta articular;
- error de seguimiento en cadera y rodilla;
- MAE y RMSE por articulación;
- torques de control;
- respuesta ante perturbaciones;
- animación o video de la simulación;
- parámetros de ejecución;
- observaciones sobre estabilidad y seguridad;
- archivos de resultados con fecha y versión.

Cuando el modelo incluya contacto pie–suelo, también se recomienda reportar:

- fuerzas de reacción del suelo;
- componentes vertical, anteroposterior y mediolateral;
- eventos de contacto;
- centro de presión, cuando el modelo lo permita;
- métricas temporales del ciclo de marcha.

---

## Reproducibilidad

Cada notebook debe registrar:

- versión del notebook;
- fecha de ejecución;
- responsable;
- sistema operativo;
- versión de Python;
- versión de MuJoCo;
- versiones de dependencias;
- archivo de modelo utilizado;
- parámetros antropométricos;
- parámetros mecánicos;
- ganancias del controlador;
- límites articulares y de torque;
- condiciones iniciales;
- perturbaciones;
- semillas aleatorias;
- métricas obtenidas;
- ruta de las evidencias generadas.

Se recomienda incluir al final de cada ejecución una tabla de configuración y resultados.

---

## Convenciones de desarrollo

### Ramas

```text
main
develop
feature/<nombre>
fix/<nombre>
docs/<nombre>
```

### Commits

Se recomienda utilizar mensajes breves y trazables:

```text
feat: agrega control por torque computado
feat: incorpora cálculo de fuerzas de reacción
fix: corrige selección de actuadores MuJoCo
docs: actualiza README institucional
test: agrega caso de marcha nominal
refactor: reorganiza parámetros del modelo
```

### Versionamiento

- `v0.x`: prueba de concepto;
- `v1.0`: línea base reproducible aprobada;
- `v1.x`: mejoras compatibles;
- `v2.0`: cambios mayores en modelo, arquitectura o interfaz.

---

## Criterios iniciales de aceptación

| Criterio | Evidencia esperada |
|---|---|
| El notebook se ejecuta de inicio a fin | Log o ejecución limpia |
| El modelo MuJoCo carga sin errores | Captura, salida de consola o prueba automatizada |
| Los parámetros están documentados | Tabla o archivo de configuración |
| Existe una simulación de marcha nominal | Figuras, video y resultados |
| Existe al menos una estrategia de control | Código, ecuaciones y métricas |
| Se reportan errores de seguimiento | MAE, RMSE y curvas |
| Los resultados son reproducibles | Entorno y semillas documentadas |
| Los supuestos clínicos están identificados | Sección de alcance y limitaciones |
| Los límites de seguridad están explícitos | Límites articulares, torque y parada |

---

## Seguridad y uso responsable

> **Advertencia:** este repositorio contiene modelos y simulaciones para investigación, formación y diseño preliminar.

Antes de trasladar cualquier estrategia de control a hardware real deben completarse, como mínimo:

- revisión clínica;
- análisis de riesgos;
- definición de límites articulares;
- límites de torque, velocidad y potencia;
- pruebas unitarias y de regresión;
- pruebas de banco sin usuario;
- parada de emergencia;
- validación de sensores;
- manejo de pérdida de comunicación;
- evaluación de saturación de actuadores;
- revisión ética y consentimiento informado;
- aprobación del protocolo correspondiente.

No deben realizarse pruebas con participantes a partir de estos notebooks sin la validación técnica, clínica y ética requerida.

---

## Relación con los entregables del proyecto

Esta carpeta aporta evidencia al entregable:

> **Repositorio de Simulación: Modelos en MATLAB/Simulink y Python.**

También sirve como insumo para:

- Informe Técnico de requerimientos de diseño mecatrónico;
- Prototipo Alfa: exoesqueleto diseñado;
- Prototipo Alfa: exoesqueleto ensamblado, integrado y funcional;
- Informe de Validación Técnica;
- Matriz de Riesgos;
- Repositorio de Procesamiento v1.0;
- Repositorio de Modelos ML;
- Repositorio de Modelos DL;
- software para el análisis de marcha;
- protocolo clínico y validación posterior.

---

## Propiedad intelectual y datos

No se deben cargar al repositorio público:

- datos clínicos identificables;
- historias clínicas;
- consentimientos informados;
- datos personales de participantes;
- credenciales;
- secretos de acceso;
- archivos con información institucional restringida;
- modelos o diseños sujetos a acuerdos de confidencialidad.

Los datos de ejemplo deben ser sintéticos, anonimizados o contar con autorización explícita para su publicación.

---

## Autores y contribuciones

El repositorio es desarrollado por investigadores, docentes y estudiantes vinculados al Proyecto G8, con participación de:

- Universidad de Antioquia;
- Universidad EAFIT;
- Institución Universitaria ITM;
- Universidad CES;
- Comité de Rehabilitación de Antioquia.

Las contribuciones individuales deben documentarse mediante historial de Git, pull requests, issues y archivos de autoría cuando corresponda.

---

## Citación

Cuando los notebooks o resultados sean utilizados en informes, presentaciones o publicaciones, se recomienda citar:

```text
Proyecto G8 – Exoesqueleto e IA para Marcha Infantil.
Repositorio de modelado, simulación y control humano–exoesqueleto.
Universidad de Antioquia, Universidad EAFIT, Institución Universitaria ITM,
Universidad CES y Comité de Rehabilitación de Antioquia, 2026.
```

Como referencia técnica complementaria para el modelado y control de exoesqueletos de miembro inferior:

```bibtex
@article{yu2023design,
  title   = {Design and Gait Control of an Active Lower Limb Exoskeleton for Walking Assistance},
  author  = {Yu, Lingzhou and Leto, Harun and Bai, Shaoping},
  journal = {Machines},
  volume  = {11},
  number  = {9},
  pages   = {864},
  year    = {2023},
  doi     = {10.3390/machines11090864}
}
```

---

## Licencia

La licencia del repositorio debe definirse de acuerdo con las políticas de propiedad intelectual de las instituciones participantes y con los compromisos establecidos en el convenio del proyecto.

Hasta que exista una licencia aprobada, el contenido debe considerarse:

> **Uso académico e institucional. Todos los derechos reservados.**

---

## Estado del desarrollo

**Versión documental:** `v0.1`  
**Estado:** línea base de simulación en construcción  
**Fecha:** julio de 2026

---

<p align="center">
  <strong>Proyecto G8 · Exoesqueleto e IA para Marcha Infantil</strong><br>
  Modelado y simulación para una rehabilitación robótica pediátrica segura, trazable y basada en evidencia.
</p>
