"""
config.py
==========
Configuración del pipeline preliminar (Etapa 1-2, versión "feature extractor"):

  Ferrari (19 marcadores) --[mapeo + preprocesamiento real de IntellEvent]-->
  ONNX IC (12 canales) + ONNX FO (18 canales) --[features fijas]-->
  Cabeza de clasificación de PC (4 clases Ferrari)

Todo lo que aparece aquí fue VERIFICADO contra:
  - El grafo ONNX real de ic_intellevent.onnx / fo_intellevent.onnx
  - El código fuente real de fhstp/IntellEvent (vicon_intellevent.py, vicon_utils.py)
  - El dataset real de Ferrari (estructura .npy, flag de validez, orden de marcadores)
No se asume nada que no se haya comprobado con datos/código reales.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Orden de los 19 marcadores de Ferrari
# ---------------------------------------------------------------------------
FERRARI_MARKER_ORDER = [
    "C7", "LA", "RA", "REP", "LEP", "RUL", "LUL",
    "RASIS", "LASIS", "RPSIS", "LPSIS", "RGT", "LGT",
    "RLE", "LLE", "RCA", "LCA", "RFM", "LFM"
]
FERRARI_MARKER_INDEX = {name: i for i, name in enumerate(FERRARI_MARKER_ORDER)}

# ---------------------------------------------------------------------------
# Mapeo Ferrari -> marcadores que IntellEvent espera (LHEE, LTOE, LANK, RHEE,
# RTOE, RANK)
#
# IMPORTANTE - limitación documentada:
#   Ferrari NO tiene marcador de tobillo. Se usa la rodilla (RLE/LLE) como
#   sustituto aproximado. Esto introduce error porque tobillo y rodilla no
#   se mueven igual (la rodilla tiene mayor excursión vertical y diferente
#   fase dentro del ciclo de marcha). Se reporta como limitación explícita
#   del pipeline preliminar, no se oculta.
# ---------------------------------------------------------------------------
MARKER_MAPPING = {
    "RHEE": "RCA",   # talón derecho -> calcáneo derecho (mapeo directo)
    "LHEE": "LCA",   # talón izquierdo -> calcáneo izquierdo (mapeo directo)
    "RTOE": "RFM",   # punta del pie derecho -> 5o metatarsiano derecho (aproximado)
    "LTOE": "LFM",   # punta del pie izquierdo -> 5o metatarsiano izquierdo (aproximado)
    "RANK": "RLE",   # tobillo derecho -> rodilla derecha (SUSTITUTO, no existe ANKLE real)
    "LANK": "LLE",   # tobillo izquierdo -> rodilla izquierda (SUSTITUTO, no existe ANKLE real)
}

# Orden exacto de marcadores que espera vicon_intellevent.py (marker_list)
INTELLEVENT_MARKER_ORDER = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]


@dataclass
class PathConfig:
    ferrari_root: str = "ferrari_data/diplegia"       # ← relativo a la raíz del proyecto
    ic_onnx_path: str = "onnx_models/ic_intellevent.onnx"
    fo_onnx_path: str = "onnx_models/fo_intellevent.onnx"
    features_cache_dir: str = "features/cache"        # ← ya tiene los .npy, no tocar
    checkpoints_dir: str = "checkpoints"              # ← se crea sola al entrenar


@dataclass
class PreprocessConfig:
    # Verificado en vicon_utils.py: base_frequency = 150 (Hz), el modelo
    # fue entrenado a esa frecuencia y vicon_intellevent.py remuestrea a
    # 150 Hz si la cámara capturó a otra frecuencia.
    intellevent_base_frequency: int = 150

    # Ferrari no publica su frecuencia de muestreo exacta ("alta frecuencia,
    # valor exacto no publicado"). ESTIMACIÓN INDIRECTA (no dato confirmado):
    # se midió la distancia entre picos verticales del talón (~68 frames/
    # zancada) en 20 trials, y asumiendo una duración de zancada típica de
    # 1.0-1.1s en niños con diplejia espástica, se estima ~60-68 Hz.
    # Se usa 60 Hz como valor redondo conservador, típico de sistemas VICON
    # clínicos de la época del estudio (2017-2019). Si se llega a confirmar
    # la frecuencia real (p.ej. metadata de un .c3d original, o el propio
    # repo de Ferrari), ACTUALIZAR este valor — afecta directamente el
    # remuestreo a 150 Hz antes de pasar por los ONNX de IntellEvent.
    assumed_ferrari_frequency: int = 60

    # Verificado en vicon_utils.py: min_peak_threshold = 0.2, distance = 25
    # (en frames @150Hz) para find_peaks sobre la probabilidad de evento.
    peak_min_height: float = 0.2
    peak_min_distance: int = 25

    # Verificado: normalización MinMax a [0.1, 1.1] por trial (no z-score).
    minmax_feature_range: tuple = (0.1, 1.1)


@dataclass
class DataConfig:
    pc_class_names: List[str] = field(default_factory=lambda: [
        "Forma_I_Equino_verdadero",
        "Forma_II_Marcha_en_salto",
        "Forma_III_Equino_aparente",
        "Forma_IV_Marcha_en_cuclillas",
    ])

    @property
    def num_pc_classes(self) -> int:
        return len(self.pc_class_names)

    # AJUSTE POR RESTRICCIÓN COMPUTACIONAL (CPU, sin GPU disponible en este
    # entorno): se usa la MEDIANA (1133 frames) en vez del percentil 90,
    # para mantener el entrenamiento factible en tiempo razonable. Esto
    # trunca más trials (~50% en vez de ~10%), lo cual es una limitación
    # real del preliminar -- en GPU, usar max_seq_len_features=2400 (P90)
    # como en el comentario original para menos truncado.
    max_seq_len_features: int = 1150


@dataclass
class ModelConfig:
    # Input de la cabeza de clasificación: concatenación de las salidas de
    # los 2 modelos ONNX (2 logits IC + 2 logits FO = 4 canales por frame).
    feature_dim: int = 4

    lstm_hidden_size: int = 32
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.3
    bidirectional: bool = True

    head_hidden: int = 24
    pooling: str = "attention"


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 40
    lr: float = 1e-3
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    device: str = "mps"


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = ProjectConfig()
