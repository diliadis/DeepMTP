from importlib.metadata import PackageNotFoundError, version

from DeepMTP.config import (
    ConfigDeprecationWarning,
    ConfigValidationError,
    DeepMTPConfig,
)
from DeepMTP.data_loading import DataLoaderFactory, TrainingDataLoaders
from DeepMTP.evaluation import EvaluationPolicy, Evaluator
from DeepMTP.hpo_progress import (
    ConsoleHPOProgressObserver,
    HPOProgressEvent,
    HPOProgressObserver,
    NullHPOProgressObserver,
)
from DeepMTP.model_factory import (
    ModelBundle,
    ModelFactory,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
)
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
    "DataLoaderFactory",
    "TrainingDataLoaders",
    "EvaluationPolicy",
    "Evaluator",
    "HPOProgressEvent",
    "HPOProgressObserver",
    "ConsoleHPOProgressObserver",
    "NullHPOProgressObserver",
    "ModelBundle",
    "ModelFactory",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
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
