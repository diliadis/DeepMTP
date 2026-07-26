from importlib.metadata import PackageNotFoundError, version

from DeepMTP.config import (
    ConfigDeprecationWarning,
    ConfigNormalizationWarning,
    ConfigValidationError,
    DeepMTPConfig,
)
from DeepMTP.data_loading import DataLoaderFactory, TrainingDataLoaders
from DeepMTP.data_progress import (
    ConsoleDataProgressObserver,
    DataProgressEvent,
    DataProgressObserver,
    NullDataProgressObserver,
)
from DeepMTP.evaluation import EvaluationPolicy, Evaluator
from DeepMTP.evaluation_progress import (
    ConsoleEvaluationProgressObserver,
    EvaluationProgressEvent,
    EvaluationProgressObserver,
    NullEvaluationProgressObserver,
)
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
    "ConfigNormalizationWarning",
    "ConfigValidationError",
    "DeepMTPConfig",
    "DataLoaderFactory",
    "TrainingDataLoaders",
    "DataProgressEvent",
    "DataProgressObserver",
    "ConsoleDataProgressObserver",
    "NullDataProgressObserver",
    "EvaluationPolicy",
    "Evaluator",
    "EvaluationProgressEvent",
    "EvaluationProgressObserver",
    "ConsoleEvaluationProgressObserver",
    "NullEvaluationProgressObserver",
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
