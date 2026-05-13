# Data directory

This directory hosts the canonical scenario data used by the simulator.

## Layout

```
data/
  test_scenarios/
    TestCases.txt                  # human-readable scenario summaries
    MealData_caseN.data            # two-column meal schedules (time<sp>carbs)
    ExerciseData_caseN.data        # two-column exercise schedules (time<sp>active)
```

Files were migrated from the legacy `TestData/` directory at the repo
root. The legacy location is no longer used; if you have custom
scripts that hardcode `TestData/`, set the environment variable
`AP_RL_DATA_DIR` to point to the directory you prefer, or run scripts
from the repo root where `ap_rl.utils.paths.data_dir()` will resolve
this layout automatically.

These scenarios are synthetic demo data, not clinical patient records.
