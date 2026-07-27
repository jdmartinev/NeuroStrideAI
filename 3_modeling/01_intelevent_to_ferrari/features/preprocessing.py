"""
features/preprocessing.py
===========================
Replica el preprocesamiento REAL de vicon_intellevent.py (repo fhstp/IntellEvent),
adaptado para recibir marcadores de Ferrari (mapeados) en vez de Vicon Nexus en vivo.

Pasos (verificados contra el código fuente original, no inventados):
  1. Detectar eje de progresión (X o Y) comparando varianza de movimiento.
  2. Construir ic_traj (2 ejes x 6 marcadores = 12) y fo_traj (3 ejes x 6 = 18).
  3. Estandarizar dirección si el sujeto camina "hacia atrás" en ese eje.
  4. Calcular velocidad con np.gradient.
  5. Normalizar con MinMax a [0.1, 1.1] por trial.
  6. Remuestrear a 150 Hz (frecuencia base de IntellEvent) si es necesario.
  7. Dar forma final (1, num_frames, num_features).

Diferencias respecto al original (documentadas, no escondidas):
  - El original lee de la API en vivo de Vicon Nexus; aquí leemos de un
    array numpy ya cargado desde Ferrari.
  - Los 6 marcadores no son los reales de Vicon Plug-in-Gait; son el mapeo
    aproximado MARKER_MAPPING definido en config.py (ver limitación del
    marcador ANKLE faltante).
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

from config import CFG, FERRARI_MARKER_INDEX, MARKER_MAPPING, INTELLEVENT_MARKER_ORDER


def extract_mapped_trajectories(ferrari_trial: np.ndarray):
    """
    Extrae las trayectorias x,y,z de los 6 marcadores mapeados de Ferrari,
    en el orden que espera IntellEvent: LHEE, LTOE, LANK, RHEE, RTOE, RANK.

    Args:
        ferrari_trial: np.ndarray (T, 19, 4) -- un trial crudo de Ferrari

    Returns:
        x_traj, y_traj, z_traj: cada uno np.ndarray (6, T)
    """
    xyz = ferrari_trial[:, :, :3]   # (T, 19, 3)
    valid = ferrari_trial[:, :, 3]  # (T, 19) -- 0=válido, -1=inválido (verificado)

    x_traj, y_traj, z_traj = [], [], []
    for marker_name in INTELLEVENT_MARKER_ORDER:
        ferrari_marker = MARKER_MAPPING[marker_name]
        idx = FERRARI_MARKER_INDEX[ferrari_marker]

        x = xyz[:, idx, 0].copy()
        y = xyz[:, idx, 1].copy()
        z = xyz[:, idx, 2].copy()

        # Enmascarar frames inválidos: poner NaN para luego interpolar,
        # en vez de dejar los [0,0,0] crudos (que distorsionarían la velocidad).
        is_invalid = valid[:, idx] == -1
        x[is_invalid] = np.nan
        y[is_invalid] = np.nan
        z[is_invalid] = np.nan

        x_traj.append(x)
        y_traj.append(y)
        z_traj.append(z)

    x_traj = np.array(x_traj)  # (6, T)
    y_traj = np.array(y_traj)
    z_traj = np.array(z_traj)

    # Interpolar frames inválidos (lineal), si los hay
    for traj in (x_traj, y_traj, z_traj):
        for i in range(traj.shape[0]):
            row = traj[i]
            nans = np.isnan(row)
            if nans.any() and not nans.all():
                idx_valid = np.where(~nans)[0]
                row[nans] = np.interp(np.where(nans)[0], idx_valid, row[idx_valid])
            elif nans.all():
                row[:] = 0.0  # marcador inválido en todo el trial -> 0 (caso degenerado)

    return x_traj, y_traj, z_traj


def detect_progression_axis(x_traj, y_traj):
    """
    Réplica de la lógica real de vicon_intellevent.py:
    compara la variación absoluta promedio en X vs Y del marcador LHEE
    (aquí: el primer marcador del array, índice 0 = LHEE mapeado) para
    decidir el eje de progresión.
    """
    prog_x = x_traj[0]  # LHEE
    prog_y = y_traj[0]
    use_x = np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y))
    return use_x, prog_x, prog_y


def build_ic_fo_trajectories(x_traj, y_traj, z_traj):
    """
    Réplica exacta de la lógica de selección de ejes de vicon_intellevent.py:
      - Si el eje de progresión es X: ic_traj = [x;z], fo_traj = [x;y;z]
      - Si el eje de progresión es Y: ic_traj = [y;z], fo_traj = [y;x;z]
    """
    use_x, prog_x, prog_y = detect_progression_axis(x_traj, y_traj)

    if use_x:
        ic_traj = np.concatenate([x_traj, z_traj])          # (12, T)
        fo_traj = np.concatenate([x_traj, y_traj, z_traj])  # (18, T)
    else:
        ic_traj = np.concatenate([y_traj, z_traj])          # (12, T)
        fo_traj = np.concatenate([y_traj, x_traj, z_traj])  # (18, T)

    # Estandarización direccional: si el sujeto "camina hacia atrás" en ese
    # eje (valores negativos al inicio), invertir el signo (réplica exacta).
    if np.any(ic_traj[0, 0:10] < 0) or np.any(ic_traj[3, 0:10] < 0):
        ic_traj[0:6, :] = (ic_traj[0:6, :] - np.mean(ic_traj[0:6, :], axis=1).reshape(6, 1)) * (-1)
        fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)

    return ic_traj, fo_traj


def resample_data(traj, sample_freq, target_freq):
    """
    Réplica exacta de resample_data() en vicon_utils.py, usando pandas
    resample + interpolación lineal.
    Args:
        traj: np.ndarray (num_features, num_frames)
    Returns:
        np.ndarray (num_resampled_frames, num_features)
    """
    # NOTA: el código original de vicon_utils.py usa el alias 'N' (nanosegundos)
    # de pandas, válido en versiones antiguas de pandas. En pandas >= 2.x ese
    # alias cambió a 'ns'. Se usa 'ns' aquí para compatibilidad, preservando
    # exactamente la misma lógica (mismo cálculo de periodo en nanosegundos).
    period = '{}ns'.format(int(1e9 / sample_freq))
    index = pd.date_range(0, periods=traj.shape[1], freq=period)
    resampled = [
        pd.DataFrame(val, index=index)
        .resample('{}ns'.format(int(1e9 / target_freq))).mean()
        for val in traj
    ]
    resampled = [np.array(r.interpolate(method='linear')) for r in resampled]
    resampled = np.concatenate(resampled, axis=1)
    return resampled


def preprocess_ferrari_trial(ferrari_trial: np.ndarray):
    """
    Pipeline completo: de un trial crudo de Ferrari (T, 19, 4) a los inputs
    finales listos para los modelos ONNX de IntellEvent.

    Returns:
        ic_input: np.ndarray (1, T_resampled, 12) listo para ic_intellevent.onnx
        fo_input: np.ndarray (1, T_resampled, 18) listo para fo_intellevent.onnx
    """
    x_traj, y_traj, z_traj = extract_mapped_trajectories(ferrari_trial)
    ic_traj, fo_traj = build_ic_fo_trajectories(x_traj, y_traj, z_traj)

    # Velocidad (derivada por diferencias finitas centradas, como el original)
    ic_velo = np.gradient(ic_traj, axis=1)
    fo_velo = np.gradient(fo_traj, axis=1)

    # Normalización MinMax [0.1, 1.1] por trial (verificado, no z-score)
    ic_velo = preprocessing.minmax_scale(ic_velo, feature_range=CFG.preprocess.minmax_feature_range, axis=1)
    fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=CFG.preprocess.minmax_feature_range, axis=1)

    # Remuestreo a 150 Hz (frecuencia base de IntellEvent) usando la
    # frecuencia ASUMIDA de Ferrari (ver limitación documentada en config.py)
    src_freq = CFG.preprocess.assumed_ferrari_frequency
    tgt_freq = CFG.preprocess.intellevent_base_frequency

    if src_freq != tgt_freq:
        ic_velo = resample_data(ic_velo, src_freq, tgt_freq).transpose()
        fo_velo = resample_data(fo_velo, src_freq, tgt_freq).transpose()

    # Forma final (1, num_frames, num_features)
    ic_input = np.transpose(ic_velo.reshape(1, ic_velo.shape[0], ic_velo.shape[1]), (0, 2, 1)).astype(np.float32)
    fo_input = np.transpose(fo_velo.reshape(1, fo_velo.shape[0], fo_velo.shape[1]), (0, 2, 1)).astype(np.float32)

    return ic_input, fo_input


if __name__ == "__main__":
    # Prueba de sanidad con un trial real
    import sys
    sys.path.insert(0, "/home/claude/nsa_v2")
    trial = np.load("/home/claude/ferrari_data/diplegia/0/10/10_1.npy")
    ic_input, fo_input = preprocess_ferrari_trial(trial)
    print("ic_input shape:", ic_input.shape, " rango:", ic_input.min(), ic_input.max())
    print("fo_input shape:", fo_input.shape, " rango:", fo_input.min(), fo_input.max())
