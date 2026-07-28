import pytest

from scripts.smoke_test_installation import (
    main,
    require_installed_wheel,
    run_smoke_test,
)


def test_installation_smoke_runs_offline(tmp_path):
    version, metrics, prediction_count = run_smoke_test(tmp_path / "results")

    assert version != "0+unknown"
    assert "accuracy_macro" in metrics
    assert prediction_count > 0
    assert (tmp_path / "results" / "wheel-smoke" / "config.json").is_file()


def test_installation_smoke_rejects_source_checkout():
    with pytest.raises(RuntimeError, match="source checkout"):
        require_installed_wheel()


def test_installation_smoke_cli_reports_result(tmp_path, capsys):
    output_dir = tmp_path / "cli-results"

    assert main(["--output-dir", str(output_dir)]) == 0
    output = capsys.readouterr().out

    assert "Verified installed DeepMTP" in output
    assert "accuracy_macro=" in output
    assert (output_dir / "wheel-smoke" / "config.json").is_file()
