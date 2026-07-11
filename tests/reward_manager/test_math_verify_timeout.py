import os

os.environ.setdefault("SANDBOX_ENDPOINT", "http://127.0.0.1:9/faas/sandbox/")

from recipe.simpletir.workers.reward_manager import math_verify
from recipe.simpletir.workers.reward_manager import math_verify_with_exec


def test_math_reward_manager_allows_ray_scheduling_headroom(monkeypatch):
    manager = math_verify.MathRewardManager(tokenizer=None, num_examine=0)
    observed_timeouts = []

    class FakeRemoteFunction:
        @staticmethod
        def remote(*args, **kwargs):
            return object()

    def fake_get(future, timeout):
        observed_timeouts.append(timeout)
        return 1.0

    monkeypatch.setattr(math_verify, "reward_func_timeout_ray", FakeRemoteFunction())
    monkeypatch.setattr(math_verify.ray, "get", fake_get)

    scores, _ = manager.math_compute_score_parallel_with_ray(
        ["deepscaler/aime"], ["\\boxed{321}"], ["321"], [None]
    )

    assert scores == [1.0]
    assert observed_timeouts == [manager.ray_get_timeout_seconds]
    assert manager.ray_get_timeout_seconds > manager.timeout_seconds


def test_math_reward_managers_define_ray_scheduling_headroom():
    managers = (
        math_verify.MathRewardManager(tokenizer=None, num_examine=0),
        math_verify_with_exec.MathRewardExecManager(tokenizer=None, num_examine=0),
    )

    for manager in managers:
        assert manager.ray_get_timeout_seconds > manager.timeout_seconds
