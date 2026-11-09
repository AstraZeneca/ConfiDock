"""Tests for the W&B helpers used by the Boltz DockQ-head runs.

Nothing here touches the network.  What is worth pinning is the behaviour that
only shows up on a six-GPU job hours in: that a missing key, an unreachable
tracker or a NaN-filled report degrades to a warning instead of taking the run
down, and that the report flattening produces the key names the charts are built
against.
"""

from __future__ import annotations

import logging

import pytest

from confidock.boltz.finetune import tracking


@pytest.fixture(autouse=True)
def _no_live_run():
    """Never leave a module-level run behind for the next test."""
    tracking._RUN = None
    yield
    tracking._RUN = None


class TestLoadWandbEnv:
    def test_parses_keys_and_strips_quotes(self, tmp_path, monkeypatch):
        env = tmp_path / "wandb_env"
        env.write_text('# a comment\n\nWANDB_API_KEY="secret"\nWANDB_ENTITY=team\n')
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        tracking.load_wandb_env(env)
        assert tracking.os.environ["WANDB_API_KEY"] == "secret"
        assert tracking.os.environ["WANDB_ENTITY"] == "team"

    def test_file_values_override_the_shell(self, tmp_path, monkeypatch):
        """A stale key exported in a shell would otherwise mis-attribute the run."""
        monkeypatch.setenv("WANDB_ENTITY", "stale")
        env = tmp_path / "wandb_env"
        env.write_text("WANDB_ENTITY=correct\n")
        tracking.load_wandb_env(env)
        assert tracking.os.environ["WANDB_ENTITY"] == "correct"

    def test_missing_file_is_not_an_error(self, tmp_path):
        tracking.load_wandb_env(tmp_path / "absent")

    def test_the_default_path_resolves_at_call_time(self, tmp_path, monkeypatch):
        """Not via a default argument, which binds ``WANDB_ENV_PATH`` at import.

        A frozen default reads the real credentials file whatever the caller says,
        so redirecting the module attribute has to work: it is how a test isolates
        itself from the developer's own key, and how a caller points at a mounted
        secret.
        """
        monkeypatch.setenv("WANDB_ENTITY", "before")
        env = tmp_path / "wandb_env"
        env.write_text("WANDB_ENTITY=redirected\n")
        monkeypatch.setattr(tracking, "WANDB_ENV_PATH", env)
        tracking.load_wandb_env()
        assert tracking.os.environ["WANDB_ENTITY"] == "redirected"


class TestInitRun:
    def test_falsy_project_disables_tracking(self):
        assert tracking.init_run(None) is False
        assert tracking.init_run("") is False

    def test_missing_api_key_does_not_raise(self, tmp_path, monkeypatch, caplog):
        """No credentials on the box must cost the charts, not the run."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setattr(tracking, "WANDB_ENV_PATH", tmp_path / "absent")
        with caplog.at_level(logging.WARNING):
            assert tracking.init_run("confidock-boltz_ft") is False
        assert "WANDB_API_KEY" in caplog.text

    def test_a_failing_tracker_is_only_a_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("WANDB_API_KEY", "x")
        monkeypatch.setattr(tracking, "WANDB_ENV_PATH", tmp_path / "absent")

        class _Boom:
            @staticmethod
            def init(**_kwargs):
                raise RuntimeError("network down")

        monkeypatch.setitem(__import__("sys").modules, "wandb", _Boom)
        with caplog.at_level(logging.WARNING):
            assert tracking.init_run("confidock-boltz_ft") is False
        assert "network down" in caplog.text


class _FakeRun:
    """Records what would have been sent."""

    def __init__(self):
        self.logged: list[tuple[dict, int | None]] = []
        self.finished = False

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


class TestLogMetrics:
    def test_no_run_is_a_silent_no_op(self):
        tracking.log_metrics({"a": 1.0})  # must not raise

    def test_forwards_metrics_and_step(self):
        run = _FakeRun()
        tracking._RUN = run
        tracking.log_metrics({"train/loss": 0.5}, step=7)
        assert run.logged == [({"train/loss": 0.5}, 7)]

    def test_drops_nans(self):
        """A metric that is constant within an entry correlates to NaN by design."""
        run = _FakeRun()
        tracking._RUN = run
        tracking.log_metrics({"good": 0.5, "bad": float("nan")})
        assert run.logged[0][0] == {"good": 0.5}

    def test_an_all_nan_payload_sends_nothing(self):
        run = _FakeRun()
        tracking._RUN = run
        tracking.log_metrics({"bad": float("nan")})
        assert run.logged == []

    def test_a_failing_log_does_not_raise(self, caplog):
        class _Boom(_FakeRun):
            def log(self, metrics, step=None):
                raise RuntimeError("dropped")

        tracking._RUN = _Boom()
        with caplog.at_level(logging.WARNING):
            tracking.log_metrics({"a": 1.0})
        assert "dropped" in caplog.text


class TestFinishRun:
    def test_finishes_and_clears(self):
        run = _FakeRun()
        tracking._RUN = run
        tracking.finish_run()
        assert run.finished
        assert tracking._RUN is None

    def test_no_run_is_a_no_op(self):
        tracking.finish_run()


class TestFlattenReport:
    @staticmethod
    def _report():
        return {
            "n_decoys": 100,
            "n_entries": 10,
            "oracle_mean_dockq": 0.8,
            "global": {"dockq_pred": 0.7, "iptm": 0.6},
            "per_entry": {"dockq_pred": 0.63, "iptm": 0.55},
            "top1_mean_dockq": {"dockq_pred": 0.5},
        }

    def test_namespaces_every_section(self):
        flat = tracking.flatten_report(self._report())
        assert flat["val/global/dockq_pred"] == pytest.approx(0.7)
        assert flat["val/per_entry/dockq_pred"] == pytest.approx(0.63)
        assert flat["val/top1_mean_dockq/dockq_pred"] == pytest.approx(0.5)

    def test_keeps_the_boltz_baselines(self):
        """Without these there is no bar on the chart for the head to clear."""
        flat = tracking.flatten_report(self._report())
        assert flat["val/per_entry/iptm"] == pytest.approx(0.55)

    def test_carries_the_scalars(self):
        flat = tracking.flatten_report(self._report())
        # ``n_entries`` is a count that is only a float because ``flatten_report``
        # casts every value for the metrics backend, so compare it as the integer it
        # is; ``oracle_mean_dockq`` is a real float and gets a tolerance.
        assert int(flat["val/n_entries"]) == 10
        assert flat["val/oracle_mean_dockq"] == pytest.approx(0.8)

    def test_prefix_separates_internal_from_external(self):
        flat = tracking.flatten_report(self._report(), prefix="external")
        assert "external/per_entry/dockq_pred" in flat
        assert not any(k.startswith("val/") for k in flat)

    def test_absent_sections_are_skipped_not_errors(self):
        assert tracking.flatten_report({"global": {}}) == {}
