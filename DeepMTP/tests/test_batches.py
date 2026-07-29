import pytest
import torch

from DeepMTP.data import (
    CustomBranchBatch,
    DenseBranchBatch,
    IDBranchBatch,
    ImageBranchBatch,
    MTPBatch,
    MTPBatchCollator,
)


def make_mtp_batch() -> MTPBatch:
    return MTPBatch(
        instance_id=torch.tensor([10, 11]),
        target_id=torch.tensor([20, 21]),
        instance_input=DenseBranchBatch(
            torch.tensor([[1.0], [2.0]]),
        ),
        target_input=IDBranchBatch(torch.tensor([0, 1])),
        score=torch.tensor([0.0, 1.0]),
    )


def test_mtp_batch_exposes_typed_inputs_and_legacy_mapping_keys():
    batch = make_mtp_batch()

    assert batch.batch_size == 2
    assert batch.instance_input.kind == "dense"
    assert batch.target_input.kind == "id"
    assert set(batch) == {
        "instance_id",
        "target_id",
        "instance_features",
        "target_features",
        "score",
    }
    assert batch["instance_features"] is batch.instance_input.values
    assert batch["target_features"] is batch.target_input.values
    assert torch.equal(batch["score"], batch.score)


def test_mtp_batch_moves_every_component_without_losing_input_types():
    original = make_mtp_batch()

    moved = original.to("cpu")

    assert isinstance(moved.instance_input, DenseBranchBatch)
    assert isinstance(moved.target_input, IDBranchBatch)
    for values in moved.values():
        assert values.device.type == "cpu"
    assert moved is not original


@pytest.mark.parametrize(
    ("batch_type", "values", "exception", "message"),
    [
        (
            DenseBranchBatch,
            torch.ones(2),
            ValueError,
            r"\[batch, features\]",
        ),
        (
            DenseBranchBatch,
            torch.ones(2, 3, dtype=torch.int64),
            TypeError,
            "floating-point",
        ),
        (
            IDBranchBatch,
            torch.ones(2, 1, dtype=torch.int64),
            ValueError,
            r"\[batch\]",
        ),
        (
            IDBranchBatch,
            torch.ones(2),
            TypeError,
            "integer",
        ),
        (
            ImageBranchBatch,
            torch.ones(2, 3, 16),
            ValueError,
            "channels",
        ),
        (
            CustomBranchBatch,
            torch.tensor(1),
            ValueError,
            "batch dimension",
        ),
    ],
)
def test_branch_batches_reject_invalid_tensor_contracts(
    batch_type,
    values,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        batch_type(values)


def test_mtp_batch_rejects_component_batch_size_mismatches():
    with pytest.raises(ValueError, match="same batch size"):
        MTPBatch(
            instance_id=torch.tensor([0, 1]),
            target_id=torch.tensor([0, 1]),
            instance_input=DenseBranchBatch(torch.ones(2, 3)),
            target_input=IDBranchBatch(torch.tensor([0])),
            score=torch.tensor([0.0, 1.0]),
        )


def test_collator_constructs_modality_specific_batches_and_float_dense_values():
    collator = MTPBatchCollator(instance_kind="dense", target_kind="id")
    items = [
        {
            "instance_id": 0,
            "target_id": 1,
            "instance_features": torch.tensor([1, 2]),
            "target_features": 1,
            "score": 0,
        },
        {
            "instance_id": 1,
            "target_id": 0,
            "instance_features": torch.tensor([3, 4]),
            "target_features": 0,
            "score": 1,
        },
    ]

    batch = collator(items)

    assert isinstance(batch.instance_input, DenseBranchBatch)
    assert isinstance(batch.target_input, IDBranchBatch)
    assert batch.instance_features.dtype == torch.float32
    assert batch.target_features.dtype == torch.int64
    assert batch.instance_features.shape == (2, 2)


def test_collator_rejects_unknown_input_kinds():
    with pytest.raises(ValueError, match="instance_kind"):
        MTPBatchCollator(instance_kind="sparse", target_kind="id")
