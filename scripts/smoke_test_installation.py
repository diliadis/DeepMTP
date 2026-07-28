"""Run a tiny offline CPU experiment against an installed DeepMTP package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np

import DeepMTP as deepmtp_package
from DeepMTP import DeepMTP, DeepMTPConfig, data_process

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PACKAGE_ROOT = _PROJECT_ROOT / "DeepMTP"


def require_installed_wheel() -> None:
    """Reject an import resolved from the source checkout."""
    package_file = deepmtp_package.__file__
    if package_file is None:
        raise RuntimeError("DeepMTP does not expose an importable package path.")
    package_path = Path(package_file).resolve()
    if package_path.is_relative_to(_SOURCE_PACKAGE_ROOT):
        raise RuntimeError(
            f"DeepMTP resolved from the source checkout at {package_path}; "
            "the smoke test requires an installed wheel."
        )


def run_smoke_test(output_dir: Path) -> tuple[str, dict[str, float], int]:
    """Train and predict once, returning version, metrics, and prediction count."""
    installed_version = distribution_version("DeepMTP")
    if installed_version == "0+unknown":
        raise RuntimeError("The installed DeepMTP distribution has no usable version.")
    if deepmtp_package.__version__ != installed_version:
        raise RuntimeError(
            "DeepMTP.__version__ does not match installed distribution metadata."
        )

    rng = np.random.default_rng(42)
    scores = (np.arange(24)[:, np.newaxis] + np.arange(4)[np.newaxis, :]) % 2
    data: dict[str, Any] = {
        "train": {
            "y": scores,
            "X_instance": rng.normal(size=(24, 3)),
            "X_target": None,
        }
    }
    train, validation, test, data_info = data_process(
        data,
        validation_setting="B",
    )

    config = DeepMTPConfig(
        validation_setting=data_info["detected_validation_setting"],
        problem_mode=data_info["detected_problem_mode"],
        general_architecture_version="dot_product",
        compute_mode="cpu",
        num_workers=0,
        train_batchsize=16,
        val_batchsize=16,
        num_epochs=1,
        metrics=["accuracy"],
        metrics_average=["macro"],
        evaluate_val=True,
        use_early_stopping=False,
        save_model=False,
        results_path=str(output_dir),
        experiment_name="wheel-smoke",
        instance_branch_architecture="MLP",
        instance_branch_input_dim=data_info["instance_branch_input_dim"],
        instance_branch_nodes_per_layer=[8],
        target_branch_architecture="MLP",
        target_branch_input_dim=data_info["target_branch_input_dim"],
        target_branch_nodes_per_layer=[8],
        embedding_size=4,
    )

    model = DeepMTP(config)
    validation_results = model.train(train, validation, test)
    test_results, predictions = model.predict(test, return_predictions=True)
    if "val_loss" not in validation_results:
        raise RuntimeError("The wheel smoke run did not produce validation loss.")
    if "accuracy_macro" not in test_results:
        raise RuntimeError("The wheel smoke run did not produce test accuracy.")
    if len(predictions) != len(test["y"]["data"]):
        raise RuntimeError(
            "The wheel smoke run returned an incomplete prediction frame."
        )
    return installed_version, test_results, len(predictions)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installation smoke test from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wheel-smoke-results"),
        help="Directory for the temporary experiment summary.",
    )
    parser.add_argument(
        "--require-wheel-install",
        action="store_true",
        help="Fail if DeepMTP resolves from the source checkout.",
    )
    arguments = parser.parse_args(argv)

    if arguments.require_wheel_install:
        require_installed_wheel()
    version, metrics, prediction_count = run_smoke_test(arguments.output_dir)
    print(
        f"Verified installed DeepMTP {version}: "
        f"{prediction_count} predictions, "
        f"accuracy_macro={metrics['accuracy_macro']:.6f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
