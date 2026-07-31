from importlib.metadata import PackageNotFoundError, version

from DeepMTP.architecture import (
    ArchitectureBranch,
    ArchitectureComponent,
    ArchitectureDescription,
    ArchitectureFusion,
    ParameterStatistics,
    describe_architecture,
    export_architecture_svg,
    format_architecture,
    print_architecture,
)
from DeepMTP.config import (
    ConfigDeprecationWarning,
    ConfigNormalizationWarning,
    ConfigValidationError,
    DeepMTPConfig,
)
from DeepMTP.data.loading import DataLoaderFactory, TrainingDataLoaders
from DeepMTP.data.preparation import data_process
from DeepMTP.data.preprocessing import (
    DataPreparationState,
    DensePreprocessingState,
    DenseScalingState,
    transform_dense_features,
)
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
from DeepMTP.integrations.huggingface import (
    FoundationLoRASpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    TokenizedFoundationSequences,
    tokenize_foundation_sequences,
)
from DeepMTP.main import DeepMTP
from DeepMTP.models import (
    CompositeEncoder,
    GraphEncoder,
    IDEmbedding,
    ModelBundle,
    ModelFactory,
    SequenceConv1DEncoder,
    SequenceGRUEncoder,
    SequenceTransformerEncoder,
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
    ReporterModelArtifact,
    ReporterPredictionTable,
    TensorBoardReporter,
    WandBReporter,
)
from DeepMTP.runtime.training import EpochOutput, TrainingEngine
from DeepMTP.runtime.wandb import WandBCheckpoint, download_wandb_checkpoint

try:
    __version__ = version("DeepMTP")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ArchitectureBranch",
    "ArchitectureComponent",
    "ArchitectureDescription",
    "ArchitectureFusion",
    "ConfigDeprecationWarning",
    "ConfigNormalizationWarning",
    "ConfigValidationError",
    "CompositeEncoder",
    "DeepMTP",
    "DeepMTPConfig",
    "DataPreparationState",
    "DataLoaderFactory",
    "TrainingDataLoaders",
    "DataProgressEvent",
    "DataProgressObserver",
    "DensePreprocessingState",
    "DenseScalingState",
    "ConsoleDataProgressObserver",
    "NullDataProgressObserver",
    "ParameterStatistics",
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
    "GraphEncoder",
    "FoundationLoRASpec",
    "FoundationModelSpec",
    "HuggingFaceSequenceEncoder",
    "ModelBundle",
    "ModelFactory",
    "SequenceConv1DEncoder",
    "SequenceGRUEncoder",
    "SequenceTransformerEncoder",
    "SparseMLP",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
    "ExperimentStore",
    "ExperimentReporter",
    "CompositeReporter",
    "NullReporter",
    "OptionalIntegrationError",
    "ReporterModelArtifact",
    "ReporterPredictionTable",
    "ProgressEvent",
    "ProgressObserver",
    "ConsoleProgressObserver",
    "NullProgressObserver",
    "RunHistory",
    "RunSummaries",
    "TensorBoardReporter",
    "TokenizedFoundationSequences",
    "EpochOutput",
    "TrainingEngine",
    "TabularEncoder",
    "TabularSchema",
    "WandBReporter",
    "WandBCheckpoint",
    "__version__",
    "data_process",
    "describe_architecture",
    "download_wandb_checkpoint",
    "export_architecture_svg",
    "format_architecture",
    "print_architecture",
    "tokenize_foundation_sequences",
    "transform_dense_features",
]
