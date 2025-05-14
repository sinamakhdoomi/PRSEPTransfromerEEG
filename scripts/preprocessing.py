#!/usr/bin/env python
"""
Reach‑and‑Grasp EEG preprocessing script
---------------------------------------
* Plain **Python ≥ 3.9**; no large‑language‑model tooling is used anywhere in the pipeline.
* Produces three NumPy files—``reach_grasp_x.npy`` (sensor epochs),
  ``reach_grasp_y.npy`` (labels), and ``reach_grasp_raw.fif`` (MNE Raw‑info)
  that downstream sLORETA localisation and classification scripts can load
  without extra metadata.

Example
~~~~~~~
$ python preprocess_reach_grasp.py \
        --data_dir /data/ReachGrasp \
        --subjects 15

All paths are relative; feel free to move the output files or rename them.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import resample
import mne
from mne.preprocessing import ICA

# ------------------------------------------------------------------
# Command‑line arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Pre‑process Reach‑and‑Grasp EEG dataset")
parser.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing the *.mat subject files")
parser.add_argument("--subjects", type=int, default=15,
                    help="Number of subjects to process (default: 15 → G01…G15)")
parser.add_argument("--out_prefix", type=str, default="reach_grasp",
                    help="Prefix for generated .npy /.fif files")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUBJECT_IDS   = [f"G{i:02d}" for i in range(1, args.subjects + 1)]       # G: gel-based electrodes, V: water-based electrodes, H: dry-electrodes
EVENT_CODE    = {503587: "palmar", 503588: "lateral", 768: "rest"}
EVENT_LABEL   = {name: idx for idx, name in enumerate(EVENT_CODE.values())}
FS            = 256         # sampling rate (Hz)
REST_WIN      = 5 * FS      # 5‑second window
ICA_COMPONENTS = 20          # used only when >11 EEG channels are present

all_x, all_y = [], []
feat_dim = None
raw_template = None          # will hold the last created Raw for .fif saving

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _append(vec: np.ndarray, label: int):
    """Add 1‑D trial ``vec`` and label to global containers, resampling if needed."""
    global feat_dim
    if feat_dim is None:
        feat_dim = vec.size
    if vec.size != feat_dim:
        vec = resample(vec, feat_dim)
    all_x.append(vec.astype(np.float32))
    all_y.append(label)

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
for sid in SUBJECT_IDS:
    fmat = DATA_DIR / f"{sid}.mat"
    if not fmat.exists():
        print(f"[skip] {fmat} not found")
        continue

    mat = sio.loadmat(fmat)
    eeg   = mat["signal"]                      # shape: (C, T)
    codes = mat["events"]["codes"][0][0].ravel()
    pos   = mat["events"]["positions"][0][0].ravel()

    # Channel labels from nested MATLAB cell array
    raw_labels = mat["header"]["channels_labels"]
    labels = [str(ch).strip()
              for arr in raw_labels.ravel()
              for sub in arr.ravel()
              for ch  in sub.ravel()]

    # Drop ocular channels
    eeg_idx   = [i for i, ch in enumerate(labels) if "EOG" not in ch]
    data      = eeg[eeg_idx]
    ch_names  = [labels[i] for i in eeg_idx]

    # Build MNE Raw (µV → V)
    info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types="eeg")
    raw  = mne.io.RawArray(data * 1e-6, info, verbose=False)

    # Save one Raw instance for downstream source localisation metadata
    raw_template = raw

    # Common average reference and 1–60 Hz linear‑phase FIR
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.filter(1, 60, fir_design="firwin", verbose=False)

    # Standard 10‑20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.pick_channels([ch for ch in raw.ch_names if ch in montage.ch_names])
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    # ICA for high‑density caps (>11 EEG channels)
    if len(eeg_idx) > 11:
        ica = ICA(n_components=ICA_COMPONENTS, method="infomax", random_state=42, verbose=False)
        ica.fit(raw)
        raw = ica.apply(raw)

    # One‑second epochs for palmar / lateral grasps
    events = np.array([[pos[i], 0, codes[i]]
                       for i in range(len(codes)) if codes[i] in EVENT_CODE])
    event_id = {EVENT_CODE[k]: k for k in EVENT_CODE}
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=1,
                        baseline=None, detrend=1, preload=True, verbose=False)

    # Five‑second rest windows
    rest_idx = np.where(codes == 768)[0]
    rest_trials = []
    for idx in rest_idx:
        start = pos[idx]
        stop  = pos[idx + 1] if idx + 1 < len(pos) else eeg.shape[1]
        for s in range(start, stop - REST_WIN + 1, REST_WIN):
            chunk = raw.get_data(start=s, stop=s + REST_WIN)
            rest_trials.append(chunk.flatten())
    rest_trials = np.array(rest_trials)

    # Balance
    n_min = min(len(epochs["palmar"]), len(epochs["lateral"]), len(rest_trials))
    for i in range(n_min):
        _append(epochs["palmar"][i].average().data.flatten(), EVENT_LABEL["palmar"])
        _append(epochs["lateral"][i].average().data.flatten(), EVENT_LABEL["lateral"])
        _append(rest_trials[i], EVENT_LABEL["rest"])

# ------------------------------------------------------------------
# Save artefacts
# ------------------------------------------------------------------
all_x = np.stack(all_x)
all_y = np.array(all_y, dtype=np.int64)
np.save(f"{args.out_prefix}_x.npy", all_x)
np.save(f"{args.out_prefix}_y.npy", all_y)
print("Saved", f"{args.out_prefix}_x.npy", f"{args.out_prefix}_y.npy")

# Raw‑info file for source localisation
if raw_template is not None:
    raw_template.save(f"{args.out_prefix}_raw.fif", overwrite=True, verbose=False)
    print("Saved", f"{args.out_prefix}_raw.fif")

print("Dataset shape:", all_x.shape, "| Label counts:", np.bincount(all_y))
