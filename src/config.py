from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class DataInfo:
    dataset_name: str = field(
        default="hp_fmri",
        metadata={"help": "Name of the dataset."}
    )
    fmri_dir: str = field(
        default = MISSING,
        metadata={"help": "Path to the raw fMRI data."}
    )
    dict_dir: str = field(
        default = MISSING,
        metadata={"help": "Path to save bi-modal dictionary."}
    )
    alias_emb_dir: str = field(
        default = MISSING,
        metadata={"help": "Path to save all word embeddings."}
    )
    word_reps_dir: str = field(
        default = MISSING,
        metadata={"help": "Path to save decontextualized word embeddings."}
    )
    fmri_reps_dir: str = field(
        default = MISSING,
        metadata={"help": "Path to save word level fmri representations."}
    )
    fmri_sentences_path: str = field(
        default = MISSING,
        metadata={"help": "All text sentences of datasets."}
    )
    num_subjects: int = field(
        default=8,
        metadata={"help": "Number of subjects."}
    )
    num_folds: int = field(
        default=4,
        metadata={"help": "Number of folds."}
    )
    tr_num: int = field(
        default=1351,
        metadata={"help": "Number of TRs."}
    )


@dataclass
class ModelInfo:
    model_type:str = field(
        default="LM",
        metadata={"help": "Type of model."}
    )
    model_id: str = field(
        default="google/bert_uncased_L-2_H-128_A-2",
        metadata={"help": "Model ID from huggingface."}
    )
    model_alias: str = field(
        default=MISSING,
        metadata={"help": "Shortcut of the model."}
    )
    model_name: str = field(
        default=MISSING,
        metadata={"help": "Specific model name."}
    )
    is_avg: bool = field(
        default=True,
        metadata={"help": "Average the representations of the tokens for one alias."}
    )
    n_layers: int = field(
        default=MISSING,
        metadata={"help": "Number of layers (including embedding layer)."}
    )
    dim: int = field(
        default=MISSING,
        metadata={"help": "Dimension of the representation."}
    )
    model_size: float = field(
        default=MISSING,
        metadata={"help": "Size of the model. (e.g. 125 for opt-125m)"}
    )

    def __post_init__(self):
        if len(self.model_id.split("/")) > 1:
            self.model_alias = self.model_id.split("/")[1].split("-")[0].split("_")[0]
            self.model_name = self.model_id.split("/")[1]
        else:
            self.model_alias = self.model_id.split("-")[0]
            self.model_name = self.model_id


@dataclass
class GaussianParams:
    vec_dim: int = field(
        default=MISSING,
        metadata={"help": "Dimension of the reperesentation from the model."}
    )
    lookforward: float = field(
        default=2.0,
        metadata={"help": "Number of seconds to look back."}
    )
    lookbackward: float = field(
        default=2.0,
        metadata={"help": "Number of seconds to look back."}
    )
    delay: float = field(
        default=6.0,
        metadata={"help": "Number of seconds delay between real response and the response fMRI recording."}
    )
    normalize: bool = field(
        default=MISSING,
        metadata={"help": "Normalize the data."}
    )


@dataclass
class MUSEParams:
    seed: int = field(
        default=42,
        metadata={"help": "Seed for reproducibility."}
    )
    normalize_embeddings: str = field(
        default="center",
        metadata={"help": "Normalize the embeddings."}
    )
    src_lang: str = field(
        default="brain",
        metadata={"help": "Source space."}
    )
    tgt_lang: str = field(
        default=ModelInfo.model_alias,
        metadata={"help": "Target space."}
    )
    n_refinement: int = field(
        default=0,
        metadata={"help": "Number of refinement iterations (0 to disable the refinement procedure)"}
    )
    dico_train: str = field(
        default=MISSING,
        metadata={"help": "Path to the training dictionary."}
    )
    dico_eval: str = field(
        default=MISSING,
        metadata={"help": "Path to the evaluation dictionary."}
    )
    src_emb: str = field(
        default=MISSING,
        metadata={"help": "Path to the source embeddings."}
    )
    tgt_emb: str = field(
        default=MISSING,
        metadata={"help": "Path to the target embeddings."}
    )
    verbose: int = field(
        default=2,
        metadata={"help": "Verbosity level."}
    )
    exp_path: str = field(
        default="",
        metadata={"help": "Path to save the experiment."}
    )
    exp_id: str = field(
        default="",
        metadata={"help": "Experiment ID."}
    )
    cuda: bool = field(
        default=True,
        metadata={"help": "Use GPU."}
    )
    export: str = field(
        default="",
        metadata={"help": "Path to save the experiment."}
    )
    emb_dim: int = field(
        default=ModelInfo.dim,
        metadata={"help": "Dimension of the embeddings."}
    )
    max_vocab: int = field(
        default=200000,
        metadata={"help": "Maximum vocabulary size."}
    )
    dico_method: str = field(
        default="csls_knn_100",
        metadata={"help": "evalutation method."}
    )
    dico_build: str = field(
        default="S2T&T2S",
        metadata={"help": "Method to build the dictionary."}
    )
    dico_threshold: int = field(
        default=0,
        metadata={"help": "Threshold for the dictionary."}
    )
    dico_max_rank: int = field(
        default=10000,
        metadata={"help": "Maximum rank for the dictionary."}
    )
    dico_min_size: int = field(
        default=0,
        metadata={"help": "Minimum generated dictionary size (0 to disable)"}
    )
    dico_max_size: int = field(
        default=0,
        metadata={"help": "Maximum generated dictionary size (0 to disable)"}
    )
    load_optim: bool = field(
        default=False,
        metadata={"help": "Load the optimization."}
    )

    def __post_init__(self):
        self.emb_dim = ModelInfo.dim


MODEL_CONFIGS = {
    "ft": ModelInfo(
        model_id="fasttext",
        model_size=0,
        n_layers=1,
        dim=300,
    ),
    "bert-tiny": ModelInfo(
        model_id="google/bert_uncased_L-2_H-128_A-2",
        model_size=4.4,
        n_layers=3,
        dim=128,
    ),
    "bert-mini": ModelInfo(
        model_id="google/bert_uncased_L-4_H-256_A-4",
        model_size=11.3,
        n_layers=5,
        dim=256,
    ),
    "bert-small": ModelInfo(
        model_id="google/bert_uncased_L-4_H-512_A-8",
        model_size=29.1,
        n_layers=5,
        dim=512,
    ),
    "bert-medium": ModelInfo(
        model_id="google/bert_uncased_L-8_H-512_A-8",
        model_size=41.7,
        n_layers=9,
        dim=512,
    ),
    "bert-base": ModelInfo(
        model_id="bert-base-uncased",
        model_size=110,
        n_layers=13,
        dim=768,
    ),
    "bert-large": ModelInfo(
        model_id="bert-large-uncased",
        model_size=340,
        n_layers=25,
        dim=1024,
    ),
    "gpt2": ModelInfo(
        model_id="gpt2",
        model_size=117,
        n_layers=13,
        dim=768,
    ),
    "gpt2-medium": ModelInfo(
        model_id="gpt2-medium",
        model_size=345,
        n_layers=25,
        dim=1024,
    ),
    "gpt2-large": ModelInfo(
        model_id="gpt2-large",
        model_size=762,
        n_layers=37,
        dim=1280,
    ),
    "gpt2-xl": ModelInfo(
        model_id="gpt2-xl",
        model_size=1542,
        n_layers=49,
        dim=1600,
    ),
    "opt-125m": ModelInfo(
        model_id="facebook/opt-125m",
        model_size=125,
        n_layers=13,
        dim=768,
    ),
    "opt-1.3b": ModelInfo(
        model_id="facebook/opt-1.3b",
        model_size=1300,
        n_layers=25,
        dim=2048,
    ),
    "opt-6.7b": ModelInfo(
        model_id="facebook/opt-6.7b",
        model_size=6700,
        n_layers=33,
        dim=4096,
    ),
    "opt-30b": ModelInfo(
        model_id="facebook/opt-30b",
        model_size=30000,
        n_layers=49,
        dim=7168,
    ),
}


@dataclass
class RunConfig:
    datasets: DataInfo = field(
        default_factory=DataInfo,
        metadata={"help": "The dataset configs."}
    )
    models: ModelInfo = field(
        default_factory=lambda: MODEL_CONFIGS["opt-30b"],
        metadata={"help": "The model configs. The key must be in MODEL_CONFIGS."}
    )
    gaussian_params: GaussianParams = field(
        default_factory=GaussianParams
    )
    muse_params: MUSEParams = field(
        default_factory=MUSEParams
    )
    projection_method: str = field(
        default="Procrustes",
        metadata={"help": "Options: Procrustes, regression."}
    )

    
    