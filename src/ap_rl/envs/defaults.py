"""Canonical default Hovorka parameter dict.

Kept in its own module so callers (Streamlit demo, smoke scripts) can
import the defaults without paying the TensorFlow import cost incurred
by :mod:`ap_rl.training.train_a2c`.
"""

from __future__ import annotations


DEFAULT_PATIENT_PARAMS: dict = {
    "BW": 75,
    "k_a1": 0.006,
    "k_a2": 0.06,
    "k_a3": 0.05,
    "k_b1": 0.003,
    "k_b2": 0.06,
    "k_b3": 0.04,
    "k_c1": 0.5,
    "V_I": 0.12,
    "t_max_I": 55,
    "k_e": 0.138,
    "F_01": 0.0097,
    "V_G": 0.16,
    "k_12": 0.066,
    "EGP_0": 0.0161,
    "AG": 1.0,
    "t_max_G": 30,
    "G_init": 10.0,
    "A_EGP": 0.05,
    "phi_EGP": -60,
    "F_peak": 1.35,
    "K_rise": 5.0,
    "K_decay": 0.01,
    "G_thresh": 9.0,
    "k_R": 0.0031,
}
