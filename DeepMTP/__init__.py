from importlib.metadata import PackageNotFoundError, version

from DeepMTP.config import (
    ConfigDeprecationWarning,
    ConfigValidationError,
    DeepMTPConfig,
)
from DeepMTP.evaluation import EvaluationPolicy, Evaluator
from DeepMTP.persistence import ExperimentStore
from DeepMTP.presentation import (
    ConsoleProgressObserver,
    NullProgressObserver,
    ProgressEvent,
    ProgressObserver,
    RunHistory,
    RunSummaries,
)
from DeepMTP.reporting import (
    CompositeReporter,
    ExperimentReporter,
    NullReporter,
    OptionalIntegrationError,
    TensorBoardReporter,
    WandBReporter,
)
from DeepMTP.training import EpochOutput, TrainingEngine

try:
    __version__ = version("DeepMTP")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ConfigDeprecationWarning",
    "ConfigValidationError",
    "DeepMTPConfig",
    "EvaluationPolicy",
    "Evaluator",
    "ExperimentStore",
    "ExperimentReporter",
    "CompositeReporter",
    "NullReporter",
    "OptionalIntegrationError",
    "ProgressEvent",
    "ProgressObserver",
    "ConsoleProgressObserver",
    "NullProgressObserver",
    "RunHistory",
    "RunSummaries",
    "TensorBoardReporter",
    "EpochOutput",
    "TrainingEngine",
    "WandBReporter",
    "__version__",
]
