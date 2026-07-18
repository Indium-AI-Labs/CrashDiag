from __future__ import annotations

import unittest

from crashdiag.verifier import CrashDiagVerifier, RewardConfig


class _State:
    def __init__(self, *, resolved: bool, healthy: bool) -> None:
        self.resolved = resolved
        self.healthy = healthy

    def health_check(self) -> dict[str, bool]:
        return {"healthy": self.healthy}


class _Fault:
    def is_resolved(self, instance: _State) -> bool:
        return instance.resolved


class _NonBooleanFault:
    def is_resolved(self, instance: _State) -> str:
        del instance
        return "false"


class _MalformedHealthState(_State):
    def health_check(self) -> dict[str, str]:
        return {"healthy": "false"}


class VerifierTests(unittest.TestCase):
    def test_default_reward_uses_state_not_trajectory_claims(self) -> None:
        verifier = CrashDiagVerifier()
        fault = _Fault()

        self.assertEqual(
            verifier.verify(
                fault,
                _State(resolved=False, healthy=True),
                trajectory={"resolved": True, "model_says": "fixed"},
            ),
            0.0,
        )
        self.assertEqual(
            verifier.verify(fault, _State(resolved=True, healthy=True)),
            1.0,
        )

        with self.assertRaises(TypeError):
            verifier.verify(_NonBooleanFault(), _State(resolved=False, healthy=False))

    def test_shaping_is_explicit_and_mechanical(self) -> None:
        verifier = CrashDiagVerifier(
            RewardConfig(enable_shaping=True, health_shaping_reward=0.1)
        )
        fault = _Fault()

        self.assertEqual(
            verifier.verify(fault, _State(resolved=False, healthy=False)),
            0.0,
        )
        self.assertEqual(
            verifier.verify(fault, _State(resolved=False, healthy=True)),
            0.1,
        )
        self.assertEqual(
            verifier.verify(fault, _MalformedHealthState(resolved=False, healthy=True)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
