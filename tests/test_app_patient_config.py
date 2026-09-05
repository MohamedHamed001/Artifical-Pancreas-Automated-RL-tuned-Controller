from app.app import _build_patient_config
from ap_rl.envs.profile_loader import PatientProfile


def test_build_patient_config_maps_profile_to_current_domain_contract() -> None:
    profile = PatientProfile(
        name="controlled",
        display_name="Controlled",
        patient_params={"BW": 72.0, "G_init": 7.0},
        patient_weight=72.0,
    )

    config = _build_patient_config(profile, seed=42)

    assert config.name == "controlled"
    assert config.params == {"BW": 72.0, "G_init": 7.0}
    assert config.body_weight_kg == 72.0
