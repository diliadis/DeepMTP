import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.check_foundation_compatibility as compatibility
from DeepMTP import FoundationModelSpec, TokenizedFoundationSequences


def _manifest_entry(**overrides):
    entry = {
        "key": "test-model",
        "domain": "test",
        "model_name": "example/test-model",
        "revision": "a" * 40,
        "samples": ["first sample", "second sample"],
        "max_length": 32,
        "pooling": "masked_mean",
        "lora_target_modules": ["query", "value"],
    }
    entry.update(overrides)
    return entry


def _write_manifest(path: Path, entries) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "models": entries}),
        encoding="utf-8",
    )
    return path


def test_reference_manifest_covers_documented_model_families():
    references = compatibility.load_reference_models()

    assert set(references) == {"esm2", "chemberta", "bert", "biomedbert"}
    assert {reference.domain for reference in references.values()} == {
        "protein",
        "chemical/SMILES",
        "general text",
        "biomedical text",
    }
    assert all(
        compatibility._FULL_COMMIT_PATTERN.fullmatch(reference.revision)
        for reference in references.values()
    )
    assert all(
        not reference.model_name.startswith("/") for reference in references.values()
    )


def test_reference_manifest_stays_synchronized_with_docs_and_workflow():
    references = compatibility.load_reference_models()
    project_root = Path(__file__).resolve().parents[2]
    guide = (project_root / "docs/source/tutorial/foundation_models.rst").read_text(
        encoding="utf-8"
    )
    workflow = (
        project_root / ".github/workflows/foundation-compatibility.yml"
    ).read_text(encoding="utf-8")

    for reference in references.values():
        assert reference.model_name in guide
        assert reference.revision in guide
        assert f"- {reference.key}" in workflow


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"schema_version": 2, "models": [_manifest_entry()]}, "schema_version"),
        (
            {
                "schema_version": 1,
                "models": [_manifest_entry(revision="main")],
            },
            "full 40-character",
        ),
        (
            {
                "schema_version": 1,
                "models": [_manifest_entry(), _manifest_entry()],
            },
            "keys must be unique",
        ),
    ],
)
def test_reference_manifest_rejects_unreproducible_entries(tmp_path, manifest, message):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(compatibility.CompatibilityManifestError, match=message):
        compatibility.load_reference_models(path)


def test_reference_check_exercises_frozen_and_lora_modes(monkeypatch):
    reference = compatibility._parse_reference(_manifest_entry())
    tokenized = TokenizedFoundationSequences(
        rows=(np.array([1, 3, 2]), np.array([1, 4, 5, 2])),
        vocabulary_size=8,
        padding_idx=0,
        spec=FoundationModelSpec(
            model_name=reference.model_name,
            revision=reference.revision,
            resolved_tokenizer_revision=reference.revision,
        ),
    )
    modes = []

    monkeypatch.setattr(
        compatibility,
        "tokenize_foundation_sequences",
        lambda values, spec: tokenized,
    )

    def run_encoder(**kwargs):
        modes.append(kwargs["train_mode"])
        return (
            {
                "output_shape": [2, 8],
                "finite": True,
                "resolved_model_revision": reference.revision,
            },
            SimpleNamespace(),
        )

    monkeypatch.setattr(compatibility, "_run_encoder", run_encoder)

    report = compatibility.check_reference_model(reference)

    assert modes == ["frozen", "lora"]
    assert report["status"] == "passed"
    assert report["reference"]["revision"] == reference.revision
    assert report["tokenization"]["lengths"] == [3, 4]
    assert set(report["modes"]) == {"frozen", "lora"}


def test_reference_check_rejects_a_mismatched_tokenizer_revision(monkeypatch):
    reference = compatibility._parse_reference(_manifest_entry())
    tokenized = TokenizedFoundationSequences(
        rows=(np.array([1, 3, 2]),),
        vocabulary_size=8,
        padding_idx=0,
        spec=FoundationModelSpec(
            model_name=reference.model_name,
            revision=reference.revision,
            resolved_tokenizer_revision="b" * 40,
        ),
    )
    monkeypatch.setattr(
        compatibility,
        "tokenize_foundation_sequences",
        lambda values, spec: tokenized,
    )

    with pytest.raises(ValueError, match="Tokenizer revision"):
        compatibility.check_reference_model(reference)


def test_compatibility_cli_writes_a_success_report(tmp_path, monkeypatch, capsys):
    manifest = _write_manifest(tmp_path / "manifest.json", [_manifest_entry()])
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        compatibility,
        "check_reference_model",
        lambda reference, output_dim: {
            "schema_version": 1,
            "status": "passed",
            "reference": {"key": reference.key},
        },
    )

    assert (
        compatibility.main(
            [
                "--model",
                "test-model",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Verified example/test-model" in capsys.readouterr().out


def test_compatibility_cli_preserves_a_structured_failure_report(
    tmp_path,
    capsys,
):
    manifest = _write_manifest(tmp_path / "manifest.json", [_manifest_entry()])
    output = tmp_path / "result.json"

    assert (
        compatibility.main(
            [
                "--model",
                "missing",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"]["category"] == "manifest"
    assert report["error"]["type"] == "CompatibilityManifestError"
    assert "Unknown model key" in capsys.readouterr().err


def test_compatibility_workflow_is_manual_network_isolated_and_uploads_reports():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/foundation-compatibility.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert 'HF_HUB_DISABLE_TELEMETRY: "1"' in workflow
    assert 'HF_HUB_DISABLE_IMPLICIT_TOKEN: "1"' in workflow
    assert "scripts/check_foundation_compatibility.py" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "if: always()" in workflow
    assert "secrets." not in workflow
