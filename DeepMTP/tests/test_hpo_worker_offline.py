from pathlib import Path

import pytest

from DeepMTP.hpo_worker import BaseWorker


def test_worker_joins_nested_result_path_without_trailing_separator(tmp_path):
    results_path = tmp_path / "nested" / "hpo"

    worker = BaseWorker(
        train=None,
        val=None,
        test=None,
        data_info=None,
        base_config={"hpo_results_path": str(results_path)},
        metric_to_optimize="loss",
    )

    project_path = Path(worker.project_name)
    assert project_path.parent == results_path
    assert project_path.is_dir()


def test_worker_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="Invalid mode"):
        BaseWorker(
            train=None,
            val=None,
            test=None,
            data_info=None,
            base_config={"hpo_results_path": str(tmp_path)},
            metric_to_optimize="loss",
            mode="unknown",
        )
