"""Read-only history analysis must not mistake a short run for convergence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_convergence.py"
spec = importlib.util.spec_from_file_location("analyze_convergence", TOOL_PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def row(epoch, loss, score, stage="continuation"):
    return {"global_epoch": epoch, "stage": stage, "average_loss": loss,
            "validation": {"classification": {"macro_f1": score, "count": 12}}}


def history(tmp_path, rows, relative="seeds/seed42/m2_history.json"):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_boundary_winner_is_not_convergence(tmp_path):
    history(tmp_path, [row(1, 0.9, 0.5, "warmup"), row(2, 0.7, 0.6, "warmup"),
                       row(3, 0.5, 0.65, "warmup"), row(4, 0.49, 0.68), row(5, 0.33, 0.71)])
    result = tool.build_analysis(tmp_path)["histories"][0]
    assert result["best_dev_global_epoch"] == 5
    assert result["best_at_run_boundary"] is True
    assert result["convergence_established"] is False
    assert result["stages"][-1]["signal"] == "dev_improving_at_stage_boundary"
    assert result["stages"][-1]["plateau_window_observed"] is False


def test_decreasing_training_loss_with_dev_regression_flags_signal(tmp_path):
    history(tmp_path, [row(4, 0.45, 0.709), row(5, 0.28, 0.708)])
    result = tool.build_analysis(tmp_path)["histories"][0]
    assert result["best_dev_global_epoch"] == 4
    assert result["best_at_run_boundary"] is False
    assert result["stages"][-1]["signal"] == "training_loss_decreases_but_dev_declines"
    assert result["convergence_established"] is False


def test_loss_jump_between_stages_is_not_a_within_stage_trend(tmp_path):
    history(tmp_path, [row(3, 0.2, 0.6, "warmup"), row(4, 2.0, 0.65)])
    result = tool.build_analysis(tmp_path)["histories"][0]
    assert len(result["stages"]) == 2
    assert all(stage["last_train_objective_delta"] is None for stage in result["stages"])
    assert all(stage["signal"] == "insufficient_within_stage_observations" for stage in result["stages"])


def test_ties_select_earlier_epoch_and_plateau_is_only_observed(tmp_path):
    history(tmp_path, [row(4, 0.4, 0.7), row(5, 0.3, 0.7), row(6, 0.2, 0.7)])
    result = tool.build_analysis(tmp_path)["histories"][0]
    assert result["best_dev_global_epoch"] == 4
    assert result["stages"][-1]["signal"] == "observed_dev_plateau_at_current_settings"
    assert result["convergence_established"] is False


def test_unrecorded_nll_remains_unavailable(tmp_path):
    history(tmp_path, [row(1, 0.9, 0.4)])
    result = tool.build_analysis(tmp_path)["histories"][0]
    assert result["dev_nll_available"] is False
    assert result["rows"][0]["dev_nll"] is None
    assert result["rows"][0]["train_verification_loss"] is None


def test_reads_separate_online_and_fixed_checkpoint_losses(tmp_path):
    value = row(4, 0.5, 0.68)
    value["loss_components"] = {"verification_nll": 0.45, "eligible_evidence_bce": 0.2,
                                "evidence_weighted_contribution": 0.05,
                                "claim_count": 24, "eligible_evidence_count": 6}
    value["diagnostics"] = {"train_probe": {"verification_nll": 0.3, "eligible_evidence_bce": 0.1,
                                            "claim_count": 12, "eligible_evidence_count": 3},
                            "dev": {"verification_nll": 0.6, "eligible_evidence_bce": 0.4,
                                    "claim_count": 12, "eligible_evidence_count": 4}}
    history(tmp_path, [value])
    result = tool.build_analysis(tmp_path)["histories"][0]
    normalized = result["rows"][0]
    assert result["dev_nll_available"] is True
    assert normalized["train_verification_loss"] == 0.45
    assert normalized["train_evidence_loss"] == 0.2
    assert normalized["train_evidence_weighted_contribution"] == 0.05
    assert normalized["train_claim_count"] == 24
    assert normalized["train_eligible_evidence_count"] == 6
    assert normalized["train_probe_nll"] == 0.3
    assert normalized["train_probe_eligible_evidence_count"] == 3
    assert normalized["dev_nll"] == 0.6
    assert normalized["dev_eligible_evidence_count"] == 4


def test_excludes_nonwinning_trials_and_deduplicates_aliases(tmp_path):
    rows = [row(1, 0.9, 0.4)]
    history(tmp_path, rows, "tuning/lambda_0.1/m2_history.json")
    history(tmp_path, [row(1, 0.8, 0.45)], "tuning/lambda_0.5/m2_history.json")
    history(tmp_path, rows, "seeds/seed42/m2_history.json")
    (tmp_path / "selected_hyperparameters.json").write_text(json.dumps({"seed": 42, "evidence_lambda": 0.1}), encoding="utf-8")
    result = tool.build_analysis(tmp_path)
    assert len(result["histories"]) == 1
    assert len(result["histories"][0]["aliases"]) == 1
    assert result["excluded_nonwinning_trials"] == ["tuning/lambda_0.5/m2_history.json"]
    assert len(tool.build_analysis(tmp_path, all_trials=True)["histories"]) == 2


def test_infers_standalone_seed_and_never_reads_test_metrics(tmp_path):
    history(tmp_path, [row(1, 0.5, 0.6)], "m2_history.json")
    (tmp_path / "experiment.json").write_text(json.dumps({"config": {"seed": 13}}), encoding="utf-8")
    (tmp_path / "m2_test_metrics.json").write_text("not json and must never be read", encoding="utf-8")
    assert tool.build_analysis(tmp_path)["histories"][0]["seed"] == 13


def test_infers_seed_from_new_convergence_plan(tmp_path):
    history(tmp_path, [row(1, 0.5, 0.6)], "m2_history.json")
    (tmp_path / "convergence_plan.json").write_text(json.dumps({"config": {"seed": 42}}), encoding="utf-8")
    assert tool.build_analysis(tmp_path)["histories"][0]["seed"] == 42


@pytest.mark.parametrize("rows", [
    [],
    [row(1, 0.5, 0.5), row(1, 0.4, 0.6)],
    [row(1, 0.5, 0.5), row(3, 0.4, 0.6)],
    [row(1, float("nan"), 0.5)],
])
def test_rejects_corrupt_or_incomplete_history(tmp_path, rows):
    history(tmp_path, rows)
    with pytest.raises(ValueError):
        tool.build_analysis(tmp_path)


def test_output_preserves_original_and_refuses_different_study(tmp_path):
    run = tmp_path / "run"
    original = history(run, [row(1, 0.5, 0.6)])
    before = original.read_bytes()
    output = tmp_path / "analysis"
    analysis = tool.build_analysis(run)
    manifest = tool.write_analysis(analysis, output)
    assert set(manifest["files"]) == {"convergence.csv", "convergence.json", "REPORT.md"}
    assert original.read_bytes() == before
    assert ",,12\n" in (output / "convergence.csv").read_text(encoding="utf-8")
    tool.write_analysis(analysis, output)
    other = tmp_path / "other"
    history(other, [row(1, 0.5, 0.6)])
    with pytest.raises(ValueError, match="different study"):
        tool.write_analysis(tool.build_analysis(other), output)
    with pytest.raises(ValueError, match="separate"):
        tool.write_analysis(analysis, run)


def test_cli_writes_small_audit_without_model_dependencies(tmp_path):
    run = tmp_path / "run"
    history(run, [row(1, 0.5, 0.6)])
    output = tmp_path / "audit"
    result = subprocess.run([sys.executable, str(TOOL_PATH), "--run", str(run), "--output", str(output)],
                            capture_output=True, text=True, check=True)
    assert "Audited 1 histories" in result.stdout
    assert json.loads((output / "convergence.json").read_text(encoding="utf-8"))["format"] == tool.FORMAT
