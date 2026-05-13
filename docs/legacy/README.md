# Legacy modules (archive)

These files are kept for historical reference only. They are **not**
imported by the active ``ap_rl`` package and may reference paths or
dependencies that no longer exist.

* ``artificial_pancreas_simulator.py`` - a stand-alone simulator that
  combined an ``InsulinSimulator`` class with a Keras model loaded from
  ``models/nn_pid_tuning_model.h5`` and a ``data/testcases/`` layout
  that was never realised in this repo. Superseded by
  :class:`ap_rl.envs.DiabetesPIDEnv`.

If you want to revive any of this code, port it against the new
``ap_rl`` package and add tests.
