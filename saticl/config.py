from typing import Any, Optional

from torch.nn import BatchNorm2d, CrossEntropyLoss, Identity, LeakyReLU, ReLU
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau

from inplace_abn import InPlaceABN, InPlaceABNSync
from pydantic import BaseSettings, Field, validator
from saticl.cli import CallableEnum, Initializer, ObjectSettings, StringEnum
from saticl.losses import (
    CombinedLoss,
    DualTanimotoLoss,
    FocalLoss,
    FocalTverskyLoss,
    KDLoss,
    TanimotoLoss,
    UnbiasedCrossEntropy,
    UnbiasedFocalLoss,
    UnbiasedFTLoss,
    UnbiasedKDLoss,
)
from saticl.metrics import F1Score, IoU, Precision, Recall


class Datasets(StringEnum):
    potsdam = "potsdam"
    vaihingen = "vaihingen"
    agrivision = "agrivision"


class ICLMethods(StringEnum):
    FT = "FT"
    LWF = "LwF"
    LWFMC = "LwF-MC"
    ILT = "ILT"
    EWC = "EwC"
    RW = "RW"
    PI = "PI"
    MIB = "MiB"


class FilteringMode(StringEnum):
    overlap = "overlap"
    no_overlap = "noov"
    split = "split"


class Optimizers(CallableEnum):
    adam = Initializer(Adam)
    adamw = Initializer(AdamW)
    sgd = Initializer(SGD, momentum=0.9)


class Schedulers(CallableEnum):
    plateau = Initializer(ReduceLROnPlateau)
    exp = Initializer(ExponentialLR, gamma=0.95)
    cosine = Initializer(CosineAnnealingLR, T_max=10)


class Losses(CallableEnum):
    crossent = CrossEntropyLoss
    focal = FocalLoss
    tversky = FocalTverskyLoss
    tanimoto = TanimotoLoss
    compl = DualTanimotoLoss
    combo = Initializer(CombinedLoss,
                        criterion_a=Initializer(CrossEntropyLoss),
                        criterion_b=Initializer(FocalTverskyLoss))


class UnbiasLosses(CallableEnum):
    crossent = UnbiasedCrossEntropy
    focal = UnbiasedFocalLoss
    tversky = UnbiasedFTLoss
    tanimoto = TanimotoLoss
    compl = DualTanimotoLoss


class Metrics(CallableEnum):
    f1 = Initializer(F1Score, ignore_index=255)
    iou = Initializer(IoU, ignore_index=255)
    precision = Initializer(Precision, ignore_index=255, reduction="macro")
    recall = Initializer(Recall, ignore_index=255, reduction="macro")


class NormLayers(CallableEnum):
    std = Initializer(BatchNorm2d)
    iabn = Initializer(InPlaceABN, activation="leaky_relu", activation_param=0.01)
    iabn_sync = Initializer(InPlaceABNSync, activation="leaky_relu", activation_param=0.01)


class ActivationLayers(CallableEnum):
    ident = Initializer(Identity)
    relu = Initializer(ReLU, inplace=True)
    lrelu = Initializer(LeakyReLU, inplace=True)


class TrainerConfig(BaseSettings):
    cpu: bool = Field(False, description="Whether to use CPU or not")
    amp: bool = Field(True, description="Whether to use mixed precision (native)")
    batch_size: int = Field(8, description="Batch size for training")
    num_workers: int = Field(8, description="Number of workers per dataloader")
    max_epochs: int = Field(100, description="How many epochs")
    monitor: Metrics = Field(Metrics.f1, description="Metric to be monitored")
    patience: int = Field(25, description="Amount of epochs without improvement in the monitored metric")
    validate_every: int = Field(1, description="How many epochs between validation rounds")
    temperature: float = Field(2.0, description="Temperature for simulated annealing, >= 1")
    temp_epochs: int = Field(20, description="How many epochs before T goes back to 1")


class OptimizerConfig(ObjectSettings):
    target: Optimizers = Field(Optimizers.adamw, description="Which optimizer to apply")
    lr: float = Field(1e-3, description="Learning rate for the optimizer")
    momentum: float = Field(0.9, description="Momentum for SGD")
    weight_decay: float = Field(1e-2, description="Weight decay for the optimizer")

    def instantiate(self, *args, **kwargs) -> Any:
        params = dict(lr=self.lr, weight_decay=self.weight_decay, **kwargs)
        if self.target == Optimizers.sgd:
            params.update(dict(momentum=self.momentum))
        return self.target(*args, **params)


class SchedulerConfig(ObjectSettings):
    target: Schedulers = Field(Schedulers.exp, description="Which scheduler to apply")

    def instantiate(self, *args, **kwargs) -> Any:
        return self.target(*args, **kwargs)


class KDConfig(ObjectSettings):
    encoder_factor: float = Field(0.0, description="Hyperparam that enables KD on the encoder (L2)")
    decoder_factor: float = Field(0.0, description="Hyperparam that enables KD on the decoder (Soft-CE)")
    unbiased: bool = Field(False, description="Enables unbiased KD loss")
    alpha: float = Field(1.0, description="Parameter to hard-ify the soft labels")

    def instantiate(self, *args, **kwargs) -> Any:
        loss_class = UnbiasedKDLoss if self.unbiased else KDLoss
        return loss_class(*args, **kwargs, alpha=self.alpha)


class CEConfig(ObjectSettings):
    unbiased: bool = Field(False, description="Enables unbiased main loss")
    name: str = Field("crossent", description="Loss name among (crossent, focal, tversky, tanimoto)")
    alpha: float = Field(0.6, description="Alpha param. for Tversky loss (0.5 for Dice)")
    beta: float = Field(0.4, description="Beta param. for Tversky loss (0.5 foor Dice)")
    gamma: float = Field(2.0, description="Gamma param. for focal loss (1.0 for standard CE)")
    aug_factor: float = Field(0.0, description="Multiplier for the augmentation invariance regularization")
    aug_layers: int = Field(1, description="How many layers to regularize, starting from the bottom")

    def instantiate(self, *args, **kwargs) -> Any:
        assert "ignore_index" in kwargs, "Ignore index required"
        assert "old_class_count" in kwargs, "Number of old classes is required for the unbiased setting"
        # update current kwargs according to the loss
        if self.name == "focal" or self.name == "tanimoto":
            kwargs.update(gamma=self.gamma)
        elif self.name == "tversky":
            kwargs.update(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        # unbiased loss is only useful from step 1 onwards
        if self.unbiased and kwargs.get("old_class_count") > 0:
            loss_class = UnbiasLosses[self.name]
        else:
            kwargs.pop("old_class_count")
            loss_class = Losses[self.name]
        return loss_class(*args, **kwargs)


class TaskConfig(BaseSettings):
    name: str = Field(required=True, description="ICL task to perform")
    step: int = Field(0, description="Which step of the ICL pipeline")
    filter_mode: FilteringMode = Field(FilteringMode.overlap, description="How to filter out images at step T")
    step_checkpoint: Optional[str] = Field(None, description="Manual override for the checkpoint of the prev.step")


class ModelConfig(BaseSettings):
    encoder: str = Field("resnet34", description="Which backbone to use (see timm library)")
    decoder: str = Field("unet", description="Which decoder to apply")
    pretrained: bool = Field(True, description="Whether to use a pretrained encoder or not")
    freeze: bool = Field(False, description="Freeze the feature extractor in incremental steps")
    output_stride: int = Field(16, description="Output stride for ResNet-like models")
    init_balanced: bool = Field(False, description="Enable background-based initialization for new classes")
    act: ActivationLayers = Field(ActivationLayers.relu, description="Which activation layer to use")
    norm: NormLayers = Field(NormLayers.std, description="Which normalization layer to use")
    dropout2d: bool = Field(False, description="Whether to apply standard drop. or channel drop. to the last f.map")
    transforms: str = Field("", description="Automatically populated by the script for tracking purposes")

    @validator("norm")
    def post_load(cls, v, values, **kwargs):
        # override activation layer to identity for activated BNs (they already include it)
        if v in (NormLayers.iabn, NormLayers.iabn_sync):
            values["act"] = ActivationLayers.ident
        return v


class SSLModelConfig(ModelConfig):
    encoder_ir: str = Field("resnet34", description="Which backbone to use for IR")
    pretext_classes: int = Field(4, description="How many classes for the SSL task, typically 4 for rotation")


class Configuration(BaseSettings):
    seed: int = Field(1337, description="Random seed for deterministic runs")
    image_size: int = Field(512, description="Size of the input images")
    in_channels: int = Field(3, description="How many input channels, 3 for RGB, 4 for RGBIR")
    channel_drop: float = Field(0.0, description="Probability to apply channel dropout")
    modality_drop: float = Field(0.0, description="Probability to apply modality dropout (drop entire RGB or IR)")
    class_weights: str = Field(None, description="Optional path to a class weight array (npy format)")
    trainer: TrainerConfig = TrainerConfig()
    # dataset options
    data_root: str = Field(required=True, description="Path to the dataset")
    dataset: Datasets = Field(Datasets.potsdam, description="Name of the dataset")
    has_val: bool = Field(False, description="True when an ad-hoc val set is present (usually False)")
    val_size: float = Field(0.1, description="Portion of train set to use for validation")
    # ML options
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    kd: KDConfig = KDConfig()
    ce: CEConfig = CEConfig()
    # method options and regularizations
    task: TaskConfig = TaskConfig(name="")
    method: ICLMethods = Field(ICLMethods.MIB, description="Which incremental learning method to use"
                                                           "(WARNING: this overrides some other parameters)")
    # logging options
    debug: bool = Field(False, description="Enables debug prints and logs")
    name: Optional[str] = Field(None, description="Identifier if the experiment, autogenerated if missing")
    output_folder: str = Field("outputs", description="Path to the folder to store outputs")
    num_samples: int = Field(1, description="How many sample batches to visualize, requires visualize=true")
    visualize: bool = Field(True, description="Turn on visualization on tensorboard")
    comment: str = Field("", description="Anything to describe your run, gets stored on tensorboard")
    version: str = Field("", description="Code version for tracking, obtained from git automatically")

    @validator("method")
    def post_load(cls, v, values, **kwargs):
        if v == ICLMethods.MIB:
            values["ce"].unbiased = True
            values["kd"].unbiased = True
            values["kd"].decoder_factor = 10
            values["model"].init_balanced = True
        else:
            raise NotImplementedError(f"Method '{v}' not yet implemented")
        return v


class SSLConfiguration(Configuration):
    model: SSLModelConfig = SSLModelConfig()
    ssl_loss: Losses = Field(Losses.crossent, description="Which loss for self supervision")


class TestConfiguration(Configuration):
    data_root: str = Field(None, description="Path to the dataset")
    store_predictions: bool = Field(False, description="Whether to store predicted images or not")
    test_on_val: bool = Field(False, description="""Sometimes datasets do not share the GT for the test set:
                                                 if you want to run a test against the validation set, set this to true""")
