# CP Gait Analysis — Transfer Learning Pipeline

## Overview

This pipeline proposes a three-stage transfer learning framework for analyzing pathological gait in children with cerebral palsy (CP). The core idea is to leverage two large VICON-based datasets as source domains to learn rich temporal representations of gait, then probe those representations on an independent held-out dataset. The pipeline deliberately avoids gait cycle normalization, preserving temporal dynamics that classical approaches discard.

---

## Stage 1 — Pretraining: IntellEvent

### Source

**IntellEvent** — Dumphart et al., *PLOS ONE* 2023  
`github.com/fhstp/IntellEvent` · `doi.org/10.1371/journal.pone.0288555`

### Training data

| Property | Value |
|---|---|
| N | 1211 patients + 61 healthy controls |
| Institution | Orthopaedic Hospital Vienna-Speising, Austria |
| Pathologies | Malrotation deformities, club foot, infantile CP (ICP), ICP with drop foot |
| Capture | VICON, overground walking |
| Access | Data not released (hospital policy) |

### Model architecture

Two independent BiLSTM models, one per event type:

- **Input:** 3 markers — HEEL, TOE, ANKLE — 3D position + velocity → **36 channels**
- **Input shape:** `(T, 36)` padded to longest trial in training set (577 frames)
- **Architecture:** Bidirectional LSTM, multiple layers
- **Output:** Per-frame probability of 3 classes — no event / left event / right event
- **Post-processing:** Peak detection with threshold = 0.01

### Labels

Dense per-frame temporal supervision:

- **IC** — Initial Contact (heel strike): detection error ≤ 5.4 ms, rate ≥ 99%
- **FO** — Foot Off (toe off): detection error ≤ 11.3 ms, rate ≥ 95%

### Key property for transfer

IntellEvent operates on **raw, unnormalized frame sequences** — not gait-cycle-normalized data. The model cannot use normalized data because gait event detection is a prerequisite for normalization (circular dependency). As a result, the pretrained encoder encodes true temporal dynamics: walking speed, cadence, stride duration, and inter-stride variability. These are precisely the features that classical 0–100% gait cycle normalization destroys.

The pretrained weights are publicly available and serve as the starting point for Stage 2.

---

## Stage 2 — Fine-tuning: Ferrari Diplegia Classification

### Source

**Lucabergamini dataset** — Ferrari, Bergamini et al., *J. Healthcare Engineering* 2019  
`github.com/lucabergamini/gait-analysis-dataset` · `doi.org/10.1155/2019/3796898`

### Training data

| Property | Value |
|---|---|
| N | 178 patients, 1139 trials |
| Diagnosis | Spastic diplegia only |
| Institution | Italian hospital |
| Capture | High-frequency VICON |
| Markers | 19 markers × (x, y, z, valid flag) |
| Format | `.npy`, variable-length sequences |
| GRF | Not available |
| Labels | Ferrari 4-class diplegia classification |

### Marker set (19 markers)

| Marker | Anatomical location |
|---|---|
| C7 | 7th cervical vertebra |
| LA, RA | Left/Right acromion |
| REP, LEP | Left/Right lateral elbow epicondyle |
| RUL, LUL | Left/Right ulna (wrist) |
| RASIS, LASIS | Right/Left anterior superior iliac spine |
| RPSIS, LPSIS | Right/Left posterior superior iliac spine |
| RGT, LGT | Right/Left greater trochanter |
| RLE, LLE | Right/Left lateral knee epicondyle |
| RCA, LCA | Right/Left calcaneus (heel) |
| RFM, LFM | Right/Left 5th metatarsal |

### Labels: Ferrari 4-class diplegia classification

Defined by Ferrari et al. (2005) from sagittal plane kinematics. All forms are spastic diplegia; they differ in the dominant joint affected and walking strategy.

| Form | Name | Clinical description | ML difficulty |
|---|---|---|---|
| I | True equinus | Ankle plantarflexed throughout stance, hips/knees extended, possible recurvatum | Easy — most distinct |
| II | Jump gait | Ankle equinus + flexed hips/knees, knee flexion in midstance (subtle) | Hard — most common (~35%), subtle midstance feature |
| III | Apparent equinus | Normal ankle ROM, excessive hip/knee flexion — looks like toe walking but cause is proximal | Medium |
| IV | Crouch gait | Excessive dorsiflexion + knee/hip flexion + scissoring | Easy — most severe |

**Class imbalance:** Form II ≈ 35% of patients. Forms can partially overlap. Hardest boundary: II vs III.

### Input adaptation (Option 2)

IntellEvent uses 3 markers × 6 channels (position + velocity) = 36 channels. Lucabergamini provides 19 markers × 4 channels (position + valid flag). To reuse pretrained weights:

**Preprocessing per trial:**
1. Load `.npy` → shape `(T, 19, 4)`
2. Extract position `xyz` → `(T, 19, 3)`
3. Compute velocity via finite differences → `(T, 19, 3)`
4. Mask invalid frames using valid flag (zero out position + velocity where `valid = 0`)
5. Concatenate position + velocity → `(T, 19, 6)`
6. Reshape → `(T, 114)`

**Model adaptation:**
- Add a new **input projection layer** `Linear(114 → 36)` trained from scratch
- Load pretrained BiLSTM weights from IntellEvent into the recurrent layers
- Add a **classification head** on top of the final hidden state

**Training in two phases:**

*Phase 1 — warm-up (~10 epochs):* Freeze BiLSTM. Train only input projection + classifier head. Allows the projection to learn the mapping from 19-marker space into the 36-channel space the BiLSTM expects.

*Phase 2 — fine-tuning (~30 epochs):* Unfreeze BiLSTM with differential learning rates:
- Input projection + classifier head: `lr = 1e-3`
- BiLSTM: `lr = 1e-5`

**Evaluation:** Leave-One-Subject-Out (LOSO) cross-validation. Each subject held out in turn; model trained on remaining 177 and tested on the held-out subject. Gives unbiased estimate of generalization.

**Loss:** Weighted cross-entropy to handle Form II class imbalance.

### Output

The fine-tuned encoder produces a **latent embedding per trial** — a fixed-dimensional vector encoding the learned representation of that patient's gait pattern. This encoder is the central transferable artifact of the pipeline.

---

## Gait Representation: Two Analysis Levels

A key design decision is whether to analyze gait at the **full trial** level or the **per-cycle** level. Both are valid and answer different questions.

### Full trial embedding

Pass the entire walking trial (all T frames) through the encoder → one embedding vector per trial.

- Captures the patient's average gait pattern
- Best for Ferrari classification and cross-dataset comparison
- Limitation: within-trial variability is collapsed into a single point

### Per-cycle embedding

Segment the trial into individual gait cycles using the IC events detected by IntellEvent (Stage 1 output). Pass each cycle through the encoder separately → one embedding vector per stride.

- Captures stride-by-stride variability
- Enables left/right asymmetry analysis — a clinically important feature in CP
- Enables within-patient consistency analysis
- No normalization required — each cycle is a raw variable-length sequence padded to the longest cycle in the batch

**The elegant byproduct:** IC and FO events are a direct output of Stage 1. Cycle segmentation is therefore not an additional preprocessing step — it falls out naturally from the pretrained model.

### Note on cycle normalization (DTW / 0–100%)

Classical gait analysis normalizes each cycle to 100 points (0–100% of the gait cycle), optionally using Dynamic Time Warping (DTW) for softer alignment. This approach:
- Enables direct comparison of joint angle curves across patients and speeds
- Is required for classical clinical metrics (peak knee flexion at X% stance)
- **Discards** walking speed, cadence, stride duration, and rhythmic variability

This pipeline uses raw-sequence embeddings as the primary representation. Normalized cycle features can be computed in parallel as a baseline to quantify how much information temporal dynamics contribute beyond cycle shape alone.

---

## Stage 3 — Embedding Analysis: SimTK (Held-Out)

### Test data

**SimTK cp-child-gait**  
`simtk.org/projects/cp-child-gait`

| Property | Value |
|---|---|
| N | 14 subjects |
| Groups | Hemiplegia n=5 (age 9.0±2.3), Diplegia n=4 (age 10.5±1.7), TD n=5 (age 8.4±1.5) |
| Institution | Different lab (UK) |
| Capture | VICON, full-body Plug-in-Gait (39 markers) |
| GRF | AMTI force plates × 2, embedded in 10m walkway |
| Format | `.c3d` |
| Labels | CP subtype (hemi/diplegia/TD) — no Ferrari forms |

### Why SimTK as test set

SimTK is the most different dataset from training data across three axes:
- **Lab shift:** Different institution, different VICON setup, different capture frequency
- **Markerset shift:** 39-marker Plug-in-Gait vs 19-marker Lucabergamini subset
- **Population shift:** Hemiplegia and TD subjects — neither present in Lucabergamini (diplegia only)

This makes it an honest probe of generalization rather than a standard held-out test.

### Input adaptation for SimTK

Map SimTK's 39-marker Plug-in-Gait to the 19-marker subset used in Stage 2, using anatomically equivalent markers where exact names differ. Compute velocities as in Stage 2 preprocessing.

### Analysis

The fine-tuned encoder (Stage 2) is frozen. No further training.

**Embedding extraction:**
1. Preprocess SimTK trials → `(T, 114)` sequences
2. Pass through frozen encoder → embedding vector per trial
3. Optionally extract per-cycle embeddings using IntellEvent IC events

**Dimensionality reduction and visualization:**
- t-SNE and UMAP on the embedding space
- Color by group: hemiplegia / diplegia / TD

**Quantitative analysis:**
- Silhouette score — how well-separated are the three groups?
- Inter-cluster vs intra-cluster distance
- Linear separability (logistic regression on embeddings, leave-one-out)

### Open question

> Do embeddings trained on Ferrari diplegia forms (Lucabergamini, Italian hospital) spontaneously separate hemiplegia / diplegia / TD subjects from a different lab and country (SimTK, UK)?

Possible outcomes and their interpretation:

| Outcome | Interpretation |
|---|---|
| Clear separation | Encoder learned a general representation of CP gait beyond the specific diplegia forms it was trained on |
| Diplegia ≈ hemiplegia, both separate from TD | Encoder captures CP vs healthy contrast but not subtype |
| No separation | Representation is lab/markerset-specific; domain adaptation needed before generalization |
| Diplegia separates from hemiplegia and TD | Encoder is diplegia-specific — useful for the classification task but not general |

All four outcomes are informative. The analysis is explicitly framed as exploratory.

---

## Domain Shift Summary

Domain shift is present across every stage transition and must be acknowledged explicitly.

| Transition | Lab | Markerset | Pathology mix |
|---|---|---|---|
| Stage 1 → Stage 2 | Vienna → Italy | 3 markers → 19 markers | Multi-pathology → diplegia only |
| Stage 2 → Stage 3 | Italy → UK | 19 markers → 39 markers (subset used) | Diplegia → hemi + diplegia + TD |

The input projection layer (Stage 2) is the primary mechanism for handling the marker shift. Lab shift is handled implicitly by fine-tuning. Population shift in Stage 3 is not corrected — it is the thing being measured.

---

## Dataset Summary

| Dataset | Stage | N | CP type | VICON markers | GRF | Labels | Access |
|---|---|---|---|---|---|---|---|
| IntellEvent | Pretraining | 1211 | ICP, ICP+drop foot + others | 3 (heel, toe, ankle) | No | IC / FO events (per frame) | Weights only |
| Lucabergamini | Fine-tuning | 178 | Spastic diplegia | 19 | No | Ferrari 4-class | Open (.npy) |
| SimTK cp-child-gait | Embedding analysis | 14 | Hemi + diplegia + TD | 39 (Plug-in-Gait) | Yes (AMTI) | CP subtype | Open (.c3d) |

---
![intelevent2ferrari](figs/intelevent_to_ferrari.png)
---

## References

1. Dumphart B, et al. (2023). Robust deep learning-based gait event detection across various pathologies. *PLOS ONE*, 18(8), e0288555. `doi.org/10.1371/journal.pone.0288555`

2. Ferrari A, Bergamini L, et al. (2019). Gait-based diplegia classification using LSTM networks. *Journal of Healthcare Engineering*, 2019, 3796898. `doi.org/10.1155/2019/3796898`

3. Bergamini L, et al. (2017). Signal processing and machine learning for diplegia classification. *ICIAP 2017*, LNCS 10485, 97–108.

4. Ferrari A, et al. (2008). The term diplegia should be enhanced. Part I: a new rehabilitation oriented classification of cerebral palsy. *European Journal of Physical and Rehabilitation Medicine*.

5. Rodda J, Graham HK. (2001). Classification of gait patterns in spastic hemiplegia and spastic diplegia. *European Journal of Neurology*, 8(s5), 98–108.
