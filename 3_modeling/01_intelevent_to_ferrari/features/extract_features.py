"""
features/extract_features.py
==============================
Corre los modelos ONNX REALES de IntellEvent (ic_intellevent.onnx,
fo_intellevent.onnx) sobre cada trial preprocesado de Ferrari, y guarda
las probabilidades de evento por frame como "features" fijas.

Esto implementa la Opción A (extractor de features fijo) acordada:
IntellEvent no se reentrena, se usa tal cual. La salida (4 valores por
frame: 2 de IC + 2 de FO) alimenta una cabeza de clasificación de PC
nueva, entrenada desde cero.

Salida por trial: np.ndarray (T, 4) = [P(no_IC), P(IC), P(no_FO), P(FO)]
"""

import os
import glob
import numpy as np
import onnxruntime as ort

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CFG
from features.preprocessing import preprocess_ferrari_trial


def load_onnx_sessions():
    ic_sess = ort.InferenceSession(CFG.paths.ic_onnx_path)
    fo_sess = ort.InferenceSession(CFG.paths.fo_onnx_path)
    return ic_sess, fo_sess


def run_intellevent_on_trial(ferrari_trial: np.ndarray, ic_sess, fo_sess):
    """
    Args:
        ferrari_trial: np.ndarray (T, 19, 4) -- trial crudo de Ferrari
    Returns:
        features: np.ndarray (T_resampled, 4) -- [P(no_IC), P(IC), P(no_FO), P(FO)]
                  Nota: T_resampled puede diferir levemente entre ic y fo por
                  redondeos del remuestreo; se recorta al mínimo común.
    """
    ic_input, fo_input = preprocess_ferrari_trial(ferrari_trial)

    ic_input_name = ic_sess.get_inputs()[0].name
    fo_input_name = fo_sess.get_inputs()[0].name

    ic_probs = ic_sess.run(None, {ic_input_name: ic_input})[0][0]  # (T_ic, 2)
    fo_probs = fo_sess.run(None, {fo_input_name: fo_input})[0][0]  # (T_fo, 2)

    min_len = min(ic_probs.shape[0], fo_probs.shape[0])
    features = np.concatenate([ic_probs[:min_len], fo_probs[:min_len]], axis=1)  # (T, 4)

    return features.astype(np.float32)


def discover_ferrari_trials(ferrari_root: str):
    """
    Recorre diplegia/{clase}/{paciente}/{paciente}_{trial}.npy y devuelve
    una lista de (filepath, clase) para todo el dataset.
    """
    pattern = os.path.join(ferrari_root, "*", "*", "*.npy")
    files = sorted(glob.glob(pattern))

    items = []
    for f in files:
        # estructura: .../diplegia/<clase>/<paciente>/<paciente>_<trial>.npy
        clase = int(f.split(os.sep)[-3])
        items.append((f, clase))

    return items


def extract_all_features(limit=None, verbose=True):
    """
    Corre el pipeline completo sobre todos (o `limit`) los trials de Ferrari.

    Returns:
        features_list: lista de np.ndarray (T_i, 4), uno por trial (longitud variable)
        labels: np.ndarray (N,) con la clase Ferrari (0-3) de cada trial
        filepaths: lista de paths, para trazabilidad
    """
    ic_sess, fo_sess = load_onnx_sessions()
    items = discover_ferrari_trials(CFG.paths.ferrari_root)

    if limit is not None:
        items = items[:limit]

    features_list, labels, filepaths, failed = [], [], [], []

    for i, (filepath, clase) in enumerate(items):
        try:
            trial = np.load(filepath)
            feats = run_intellevent_on_trial(trial, ic_sess, fo_sess)
            features_list.append(feats)
            labels.append(clase)
            filepaths.append(filepath)
        except Exception as e:
            failed.append((filepath, str(e)))

        if verbose and (i + 1) % 50 == 0:
            print(f"  procesados {i+1}/{len(items)} trials ({len(failed)} fallidos hasta ahora)")

    if verbose:
        print(f"\nTotal procesados exitosamente: {len(features_list)} / {len(items)}")
        if failed:
            print(f"Trials fallidos: {len(failed)}")
            for fp, err in failed[:5]:
                print(f"  {fp}: {err}")

    return features_list, np.array(labels), filepaths


def save_features_cache(features_list, labels, filepaths, cache_dir):
    """Guarda las features extraídas en disco para no recalcular cada vez."""
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        os.path.join(cache_dir, "ferrari_features.npz"),
        labels=labels,
        filepaths=np.array(filepaths),
        n_trials=len(features_list),
    )
    # Las secuencias tienen longitud variable -> se guardan por separado
    for i, feats in enumerate(features_list):
        np.save(os.path.join(cache_dir, f"trial_{i:04d}.npy"), feats)
    print(f"Features guardadas en: {cache_dir}")


def load_features_cache(cache_dir):
    meta = np.load(os.path.join(cache_dir, "ferrari_features.npz"), allow_pickle=True)
    n_trials = int(meta["n_trials"])
    labels = meta["labels"]
    filepaths = meta["filepaths"]

    features_list = [
        np.load(os.path.join(cache_dir, f"trial_{i:04d}.npy"))
        for i in range(n_trials)
    ]
    return features_list, labels, filepaths


def extract_batch_resumable(cache_dir, batch_size=150, verbose=True):
    """
    Procesa el dataset completo en lotes, guardando progreso incremental en
    un archivo de estado (progress.txt) para poder reanudar si se interrumpe
    (necesario porque el dataset completo toma ~45-55 min, más que el límite
    de una sola ejecución de herramienta).
    """
    os.makedirs(cache_dir, exist_ok=True)
    progress_path = os.path.join(cache_dir, "progress.txt")

    ic_sess, fo_sess = load_onnx_sessions()
    items = discover_ferrari_trials(CFG.paths.ferrari_root)
    total = len(items)

    start_idx = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            start_idx = int(f.read().strip())

    if verbose:
        print(f"Total trials: {total}. Reanudando desde índice {start_idx}.")

    end_idx = min(start_idx + batch_size, total)
    failed = []

    for i in range(start_idx, end_idx):
        filepath, clase = items[i]
        try:
            trial = np.load(filepath)
            feats = run_intellevent_on_trial(trial, ic_sess, fo_sess)
            np.save(os.path.join(cache_dir, f"trial_{i:04d}.npy"), feats)
            with open(os.path.join(cache_dir, "labels.txt"), "a") as lf:
                lf.write(f"{i}\t{clase}\t{filepath}\n")
        except Exception as e:
            failed.append((i, filepath, str(e)))

        if verbose and (i + 1) % 25 == 0:
            print(f"  procesados {i+1}/{total}")

    with open(progress_path, "w") as f:
        f.write(str(end_idx))

    if verbose:
        print(f"Lote completado: índices [{start_idx}, {end_idx}) de {total}.")
        if failed:
            print(f"  {len(failed)} fallidos en este lote:")
            for idx, fp, err in failed:
                print(f"    [{idx}] {fp}: {err}")

    return end_idx, total, failed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "batch"], default="sanity")
    parser.add_argument("--batch_size", type=int, default=150)
    args = parser.parse_args()

    if args.mode == "sanity":
        # Prueba de sanidad: solo 5 trials.
        print("=== Prueba de sanidad: 5 trials ===")
        feats_list, labels, paths = extract_all_features(limit=5)
        for i, (f, lbl, p) in enumerate(zip(feats_list, labels, paths)):
            print(f"  trial {i}: shape={f.shape}  clase={lbl}  rango=[{f.min():.3f}, {f.max():.3f}]  archivo={p}")
    else:
        end_idx, total, failed = extract_batch_resumable(
            CFG.paths.features_cache_dir, batch_size=args.batch_size
        )
        print(f"\nProgreso: {end_idx}/{total}")
        if end_idx < total:
            print("Quedan trials por procesar. Vuelve a correr este script (--mode batch) para continuar.")
        else:
            print("Dataset completo procesado.")
