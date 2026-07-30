from importlib.metadata import PackageNotFoundError, version

from DeepMTP.config import (
    ConfigDeprecationWarning,
    ConfigNormalizationWarning,
    ConfigValidationError,
    DeepMTPConfig,
)
from DeepMTP.data.loading import DataLoaderFactory, TrainingDataLoaders
from DeepMTP.data.preparation import data_process
from DeepMTP.data.progress import (
    ConsoleDataProgressObserver,
    DataProgressEvent,
    DataProgressObserver,
    NullDataProgressObserver,
)
from DeepMTP.data.tabular import TabularSchema
from DeepMTP.hpo.progress import (
    ConsoleHPOProgressObserver,
    HPOProgressEvent,
    HPOProgressObserver,
    NullHPOProgressObserver,
)
from DeepMTP.main import DeepMTP
from DeepMTP.models import (
    IDEmbedding,
    ModelBundle,
    ModelFactory,
    SparseMLP,
    TabularEncoder,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
)
from DeepMTP.runtime.evaluation import EvaluationPolicy, Evaluator
from DeepMTP.runtime.evaluation_progress import (
    ConsoleEvaluationProgressObserver,
    EvaluationProgressEvent,
    EvaluationProgressObserver,
    NullEvaluationProgressObserver,
)
from DeepMTP.runtime.persistence import ExperimentStore
from DeepMTP.runtime.presentation import (
    ConsoleProgressObserver,
    NullProgressObserver,
    ProgressEvent,
    ProgressObserver,
    RunHistory,
    RunSummaries,
)
from DeepMTP.runtime.reporting import (
    CompositeReporter,
    ExperimentReporter,
    NullReporter,
    OptionalIntegrationError,
    TensorBoardReporter,
    WandBReporter,
)
from DeepMTP.runtime.training import EpochOutput, TrainingEngine

try:
    __version__ = version("DeepMTP")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ConfigDeprecationWarning",
    "ConfigNormalizationWarning",
    "ConfigValidationError",
    "DeepMTP",
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
    "IDEmbedding",
    "ModelBundle",
    "ModelFactory",
    "SparseMLP",
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
    "TabularEncoder",
    "TabularSchema",
    "WandBReporter",
    "__version__",
    "data_process",
]
