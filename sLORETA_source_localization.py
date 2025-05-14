#!/usr/bin/env python
"""
sLORETA projection for single‑trial EEG epochs.

Requires: MNE‑Python, NumPy.

Inputs
------
--sensor_file :  N×(C·T) NumPy array  (flattened sensor‑space trials)
--label_file  :  N‑vector NumPy array (integer class labels)
--info_file   :  Raw‑info .fif file that stores channel metadata
                 (the Raw object itself is not needed)

Outputs
-------
For every canonical band (Delta, …, Gamma) two *.npy files are saved:
<Band>_source_data.npy   (N×F)  source‑space trials
<Band>_source_labels.npy (N)    integer labels
"""

import argparse
from pathlib import Path
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.filter import filter_data

# ------------------------------------------------------------------
# Command‑line arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Single‑trial sLORETA projection")
parser.add_argument("--sensor_file", required=True, type=str,
                    help="NumPy file holding flattened sensor‑space trials")
parser.add_argument("--label_file", required=True, type=str,
                    help="NumPy file holding integer labels")
parser.add_argument("--info_file",  required=True, type=str,
                    help=".fif file that stores Raw‑info (channel names, sfreq)")
parser.add_argument("--out_dir",    default=".", type=str,
                    help="Output directory (default: current)")
args = parser.parse_args()
OUT_DIR = Path(args.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
all_data   = np.load(args.sensor_file)   # shape (N, C·T)
all_labels = np.load(args.label_file)    # shape (N,)
raw_info   = mne.io.read_info(args.info_file)  # only channel metadata

n_trials, flat_dim = all_data.shape
n_channels = len(raw_info.ch_names)
n_times    = flat_dim // n_channels
sfreq      = raw_info["sfreq"]

assert flat_dim % n_channels == 0, "C·T dimensions mismatch"

print(f"Trials: {n_trials} | Channels: {n_channels} | Time points: {n_times}")

# ------------------------------------------------------------------
# Head model (sample subject bundled with MNE)
# ------------------------------------------------------------------
subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
subject = "sample"
src  = mne.setup_source_space(subject, spacing="oct5",
                              subjects_dir=subjects_dir, add_dist=False)
bem  = mne.make_bem_solution(
           mne.make_bem_model(subject=subject, ico=4,
                              conductivity=[0.3, 0.006, 0.3],
                              subjects_dir=subjects_dir))

print("Computing forward solution …")
trans = "fsaverage"                       # generic alignment
fwd   = mne.make_forward_solution(raw_info, trans=trans, src=src,
                                  bem=bem, eeg=True, mindist=5.0)

print("Computing inverse operator (sLORETA) …")
# A dummy noise covariance (identity) is sufficient for single‑trial export
noise_cov = mne.make_ad_hoc_cov(raw_info, std=1e-6)
inverse_op = make_inverse_operator(raw_info, fwd, noise_cov,
                                   loose=1.0, depth=0.8)

# ------------------------------------------------------------------
# Frequency bands
# ------------------------------------------------------------------
FREQ_BANDS = dict(Delta=(0.5, 4),
                  Theta=(4, 8),
                  Alpha=(8, 13),
                  Beta =(14, 30),
                  Gamma=(30, 50))

snr      = 3.0
lambda2  = 1.0 / snr**2
method   = "sLORETA"

# ------------------------------------------------------------------
# Loop over bands
# ------------------------------------------------------------------
for band, (l_freq, h_freq) in FREQ_BANDS.items():
    print(f"\n[{band}] {l_freq}–{h_freq} Hz")

    src_trials = []
    for i in range(n_trials):
        # Reshape trial -> (C, T)
        trial = all_data[i].reshape(n_channels, n_times)

        # Band‑pass filter
        trial_filt = filter_data(trial.astype(np.float64), sfreq,
                                 l_freq, h_freq, method="iir", verbose=False)

        # Single‑trial Evoked
        evoked = mne.EvokedArray(trial_filt, raw_info, tmin=0.)

        # sLORETA projection
        stc = apply_inverse(evoked, inverse_op, lambda2, method=method,
                            pick_ori="normal")
        src_trials.append(stc.data.flatten())

    src_trials = np.asarray(src_trials, dtype=np.float32)

    # Save
    np.save(OUT_DIR / f"{band}_source_data.npy",   src_trials)
    np.save(OUT_DIR / f"{band}_source_labels.npy", all_labels)
    print(f"  saved → {band}_source_data.npy ({src_trials.shape})")

print("\nDone – source‑space arrays written to", OUT_DIR)
