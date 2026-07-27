import pytest

from DeepMTP import (
    ConsoleDataProgressObserver,
    DataProgressEvent,
    NullDataProgressObserver,
)


def test_console_data_progress_renders_basic_and_developer_messages(capsys):
    basic = ConsoleDataProgressObserver()
    developer = ConsoleDataProgressObserver("dev")

    basic.on_event(DataProgressEvent("Processing... ", end=""))
    basic.on_event(DataProgressEvent("Done"))
    developer.on_event(DataProgressEvent("Incomplete data", level="warning"))
    developer.on_event(DataProgressEvent("Invalid data", level="error"))
    developer.on_event(DataProgressEvent(""))

    assert capsys.readouterr().out == (
        "Processing... Done\nwarning: Incomplete data\nerror: Invalid data\n\n"
    )


def test_null_data_progress_is_silent(capsys):
    observer = NullDataProgressObserver()

    observer.on_event(DataProgressEvent("Processing..."))
    observer.on_event(DataProgressEvent("Invalid data", level="error"))

    assert capsys.readouterr().out == ""


def test_data_progress_value_preserves_existing_positional_fields():
    event = DataProgressEvent(
        "Processing... ",
        "operation_started",
        "interaction_format",
        "warning",
        "",
        "triplets",
    )

    assert event.level == "warning"
    assert event.end == ""
    assert event.value == "triplets"


@pytest.mark.parametrize("print_mode", ["unknown", [], None])
def test_console_data_progress_rejects_invalid_print_mode(print_mode):
    with pytest.raises(ValueError, match="print_mode"):
        ConsoleDataProgressObserver(print_mode)
