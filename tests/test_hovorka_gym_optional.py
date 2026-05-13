"""Optional smoke test for :class:`ap_rl.envs.hovorka_gym_env.HovorkaGymEnv`."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

from ap_rl.envs.hovorka_gym_env import HovorkaGymEnv


def test_hovorka_gym_reset_step() -> None:
    env = HovorkaGymEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (13,)
    obs2, reward, term, trunc, info2 = env.step(env.action_space.sample())
    assert obs2.shape == (13,)
    assert isinstance(reward, float)
    assert term in (True, False)
    assert trunc is False
    assert "glucose" in info2
