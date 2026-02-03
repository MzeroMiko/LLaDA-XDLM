import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import IterableDataset, IterableDatasetDict, load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import argparse
from peft import LoraConfig, get_peft_model, TaskType
import random
import numpy as np
import torch.nn.functional as F
from transformers import DefaultDataCollator
from tqdm import tqdm
import pickle
import torch.distributed as dist

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class XDMHelper:
    @staticmethod
    def forward_process(
        batch: torch.Tensor, 
        alpha_t: torch.Tensor, 
        k1: float=0.1,
        mask_id: int = -1,
        vocab_size: int = -1,
        generator=None,
    ):
        b, l = batch.shape
        alpha_t = alpha_t.view(b, 1)
        with_noise = k1 > 0.0
        with_mask = k1 < 1.0
        device = batch.device

        rand = torch.rand((b, l), device=device, generator=generator)
        transfer_id = torch.randint(0, vocab_size, (b, l), dtype=torch.int64, device=device, generator=generator)
        to_keep = rand < alpha_t

        if with_mask and with_noise:
            assert mask_id > 0
            assert vocab_size > 0
            to_noise = (~to_keep) & (rand < alpha_t + k1 * (1 - alpha_t))
            noisy_batch = torch.where(to_keep, batch, mask_id)
            noisy_batch = torch.where(to_noise, transfer_id, noisy_batch)
        elif with_mask:
            assert mask_id > 0
            noisy_batch = torch.where(to_keep, batch, mask_id)
        elif with_noise:
            assert vocab_size > 0
            noisy_batch = torch.where(to_keep, batch, transfer_id)
        return noisy_batch

    @staticmethod
    def get_kl_simp(
        logits: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask_id: Optional[int] = None,
        k1: float = 0.1,
        alpha_t: torch.Tensor = 0.1,
    ):
        """
        delta_alpha_scale = 1
        alpha_s = alpha_t
        """
        b, l, v = logits.shape
        b, l = inputs.shape
        b, l = labels.shape
        k2 = 1 - k1
        alpha_t = alpha_t.view(b, 1, 1)
        beta_t = 1 - alpha_t

        zt_eq_x = (inputs == labels).unsqueeze_(-1)
        zt_eq_m = None
        if mask_id is None:
            assert k1 == 1.0
            vratio = k1 + torch.zeros((b, l, 1), dtype=logits.dtype, device=inputs.device)
        else:
            zt_eq_m = (inputs == mask_id).unsqueeze_(-1)
            vratio = (k1 + v * k2 * zt_eq_m)
            # set probs of mask to be 0
            logits[:, :, mask_id] = torch.finfo(logits.dtype).min

        probs = logits.softmax(dim=-1)
        prob_x0 = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)) # (b, l, 1)
        prob_zt = torch.gather(probs, dim=-1, index=inputs.unsqueeze(-1)) # (b, l, 1)
        vfprob_t_x0 = alpha_t * v * prob_x0 + beta_t * k1
        vfprob_t_zt = alpha_t * v * prob_zt + beta_t * vratio
        vfhard_t_x0 = alpha_t * v + beta_t * k1
        vfhard_t_zt = alpha_t * v * zt_eq_x + beta_t * vratio
        rdivt = torch.where(zt_eq_x, k1 / vfhard_t_x0, 1 / beta_t) # r(z_t) / f(t, x, z_t)


        """
        partx = mean(log(f(s, x, e) / f(s, x_th, e)))
        mean(log (vf(s, x, e))) + partx_0 - 1/N * (-(v * alpha_t * probs) + beta_t * N * (1 - k1)).log()
        """
        if k1 > 0:
            partx_0 = - (v * alpha_t * probs + beta_t * k1).log().mean(dim=-1, keepdim=True)
            partx_1 = (beta_t * k1).log() * (v - 1) / v
            partx_2 = (v * alpha_t + beta_t * k1).log() / v 
            partx = partx_0 + partx_1 + partx_2

        kl = rdivt * (
            v * (zt_eq_x.float() - prob_zt) / vfprob_t_zt
            - 1 / alpha_t * (vfhard_t_zt.log() - vfprob_t_zt.log())
            + (vfhard_t_x0.log() - vfprob_t_x0.log())
            + (k1 * beta_t * partx / alpha_t if k1 > 0 else 0)
        )
        kl = kl.view(b, l)

        return kl


def forward_process_mdm(batch: torch.Tensor, mask_token_id=32000, eps=1e-3, prompt_length: torch.Tensor=None):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l) # ratio

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, mask_token_id, batch)
    
    if prompt_length is not None:
        prompt_mask = torch.arange((1, l), device=batch.device) < prompt_length.view(-1, 1)
        noisy_batch[prompt_mask] = batch[prompt_mask]
        mask_indices[prompt_mask] = False
    
    return noisy_batch, mask_indices, p_mask


def forward_process_xdm(batch: torch.Tensor, k1=0.1, mask_token_id=-1, vocab_size=-1, eps=1e-3, prompt_length: torch.Tensor=None):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    alpha_t = 1 - p_mask
    noisy_batch = XDMHelper.forward_process(batch, alpha_t, k1, mask_token_id, vocab_size, generator=None)
    
    if prompt_length is not None:
        prompt_mask = torch.arange((1, l), device=batch.device) < prompt_length.view(-1, 1)
        noisy_batch[prompt_mask] = batch[prompt_mask]
    
    return noisy_batch, None, alpha_t


class dllm_Trainer(Trainer):
    reverse_process_training_method = "ar"
    forward_process_mask_token_id = 126336
    forward_process_vocab_size = 126464
    forward_process_k1 = 0.1
    random_select_length_ratio = -1.0
    
    def compute_loss_mdm(self, model, input_ids, prompt_length):
        labels = input_ids
        noisy_input, mask_indices, p_mask = forward_process_mdm(
            input_ids, 
            mask_token_id=self.forward_process_mask_token_id, 
            prompt_length=prompt_length,
        )
        logits = model(
            input_ids=noisy_input
        ).logits.contiguous()

        loss = torch.nn.functional.cross_entropy(
            logits[mask_indices],
            labels[mask_indices],
            reduction="none",
        ) / p_mask[mask_indices]
        return loss.sum(), logits

    def compute_loss_xdm(self, model, input_ids, prompt_length): 
        noisy_input, _, alpha_t = forward_process_xdm(
            input_ids,
            k1=self.forward_process_k1,
            mask_token_id=self.forward_process_mask_token_id, 
            vocab_size=self.forward_process_vocab_size,
            prompt_length=prompt_length,
        )
        
        logits = model(
            input_ids=noisy_input
        ).logits.contiguous()

        loss = XDMHelper.get_kl_simp(
            logits, 
            inputs=noisy_input, 
            labels=input_ids, 
            k1=self.forward_process_k1,
            mask_id=self.forward_process_mask_token_id, 
            alpha_t=alpha_t,
        )

        if prompt_length is not None:
            prompt_mask = torch.arange((1, input_ids.shape[-1]), device=logits.device) < prompt_length.view(-1, 1)
            loss[prompt_mask] = 0
        return loss.sum(), logits

    def compute_loss_ar(self, model, input_ids, prompt_length):
        labels = input_ids
        logits = model(
            input_ids=input_ids,
            is_causal=True,
        ).logits.contiguous()

        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            reduction="none",
        )

        if prompt_length is not None:
            prompt_mask = torch.arange((1, input_ids.shape[-1]), device=logits.device) < (prompt_length - 1).view(-1, 1)
            loss[prompt_mask] = 0
        return loss.sum(), logits

    def compute_loss_core(self, model, input_ids, prompt_length):
        if self.reverse_process_training_method == "mdm":
            return self.compute_loss_mdm(model, input_ids, prompt_length)
        elif self.reverse_process_training_method == "xdm":
            return self.compute_loss_xdm(model, input_ids, prompt_length)
        elif self.reverse_process_training_method == "ar":
            return self.compute_loss_ar(model, input_ids, prompt_length)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        TODO:
            Do we need to use label_smoother ?
            we do not accept input_ids with paddings for now !!!
            we do not know whether other dlms still support AR...
            so this code is only for plada_qwen2 for now.
        """
        if not hasattr(self.args, "average_tokens_across_devices"):
            self.args.average_tokens_across_devices = False


        input_ids = inputs["input_ids"]
        prompt_length = inputs.pop("prompt_length", None)
        if self.random_select_length_ratio > 0:
            uniform_length_mask = torch.rand((1,), device=input_ids.device) < self.random_select_length_ratio
            uniform_length = (torch.rand((1,), device=input_ids.device) * (input_ids.shape[-1] - 1)).int()

            master_proc = True
            if torch.distributed.is_initialized():
                ulen = [uniform_length.clone() for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(ulen, uniform_length)
                uniform_length = torch.stack(ulen).mean(0, dtype=torch.float).int()
                master_proc = (torch.distributed.get_rank() == 0)
            
            if uniform_length_mask:
                input_ids = input_ids[:, :(uniform_length + 1)]
                if master_proc:
                    print("uniform length: ", uniform_length + 1, flush=True)

        loss, logits = self.compute_loss_core(model, input_ids, prompt_length)

        assert self.label_smoother is None
        assert not return_outputs

        if prompt_length is not None:
            # we dont support self.args.average_tokens_across_devices here
            loss = loss / (input_ids.shape[0] * input_ids.shape[-1] - prompt_length.sum())
        else:
            if self.args.average_tokens_across_devices and (num_items_in_batch is not None):
                loss = loss / (num_items_in_batch * input_ids.shape[-1])
            else:
                loss = loss / (input_ids.shape[0] * input_ids.shape[-1])

        outputs = dict(
            loss=loss,
            logits=logits,
        )

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    with_lora: bool = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        },
    )
    training_method: str = field(
        default="ar",
        metadata={
            "help": (
                ""
            )
        },
    )
    random_select_length_ratio: float = field(
        default=-1.0,
        metadata={
            "help": (
                ""
            )
        },
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def split_streaming_dataset(
    full_streaming_dataset,
    validation_percentage: int = 5,
) -> IterableDatasetDict:
    """
    Splits a streaming dataset into
    training and validation IterableDatasets, and supports methods like .map(), .filter(),
    .take() and properties like .features on the resulting streams.

    Args:
        full_streaming_dataset (Dataset): The name of the dataset to load (e.g., "HuggingFaceFW/fineweb").
        validation_percentage (int): The proportion of the dataset to be used for validation split.

    Returns:
        IterableDatasetDict: An IterableDatasetDict containing two IterableDataset objects: (train_stream, validation_stream).
    """
    if not (0 < validation_percentage < 100):
        raise ValueError(
            f"validation_percentage must be between 0 and 100 (exclusive). Passed: {validation_percentage}"
        )

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            if is_train:
                if i % 100 > validation_percentage:
                    yield example
            else:
                if i % 100 < validation_percentage:
                    yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(
        split_generator, gen_kwargs={"is_train": False}, features=features
    )

    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})


def get_args():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def detect_last_checkpoint(training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def build_dataset(model_args, data_args):
    trust_remote_code_key=dict(trust_remote_code=model_args.trust_remote_code)
    trust_remote_code_key=dict()
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            **trust_remote_code_key,
        )
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    **trust_remote_code_key,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    **trust_remote_code_key,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    **trust_remote_code_key,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    extension,
                    data_files=data_files,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.
    # raise ValueError(f"{len(lm_datasets['train'])} {len(lm_datasets['validation'])}")

    return raw_datasets


def build_model(model_args):
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resize embedding from {embedding_size} to {len(tokenizer)}")
    
    if model_args.with_lora:
        model = get_peft_model(model, LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", 
                "o_proj", "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )).to(torch.bfloat16)
        model.print_trainable_parameters()
        logger.info(f"Build lora.")
        
    
    return model, tokenizer, config


def preprocess_data(raw_datasets, tokenizer, config, model_args, data_args, training_args):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        if hasattr(config, "max_position_embeddings"):
            max_pos_embeddings = config.max_position_embeddings
        else:
            # Define a default value if the attribute is missing in the config.
            max_pos_embeddings = 1024
        
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        result.pop("labels", None)
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
    
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        train_dataset = train_dataset.shuffle(buffer_size=10000, seed=training_args.seed)
        if data_args.max_train_samples is not None:
            if data_args.streaming:
                train_dataset = train_dataset.take(data_args.max_train_samples)
            else:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    preprocess_logits_for_metrics = None
    compute_metrics = None
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            if data_args.streaming:
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
      
    return train_dataset, eval_dataset, preprocess_logits_for_metrics, compute_metrics


# modified from transformers.
def main():
    model_args, data_args, training_args = get_args()
    setup_logging(training_args)
    last_checkpoint = detect_last_checkpoint(training_args)
    ##############
    # Set seed before initializing model.
    ##############
    set_seed(training_args.seed)
    raw_datasets = build_dataset(model_args, data_args)
    model, tokenizer, config = build_model(model_args)
    (
        train_dataset, eval_dataset, 
        preprocess_logits_for_metrics, compute_metrics
    ) = preprocess_data(
        raw_datasets, tokenizer, config, 
        model_args, data_args, training_args
    )

    # Initialize our Trainer
    """
    as we can not output labels in inputs, the eval can not be used.
    """
    class _dllm_Trainer(dllm_Trainer):
        reverse_process_training_method = model_args.training_method
        random_select_length_ratio = model_args.random_select_length_ratio # fix 1130

    print(_dllm_Trainer.__dict__)
    trainer_tokenizer=dict(processing_class=tokenizer,)
    trainer_tokenizer=dict(tokenizer=tokenizer,)
    
    trainer = _dllm_Trainer(
        model=model,
        args=training_args,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        **trainer_tokenizer,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        if data_args.streaming:
            metrics["train_samples"] = max_train_samples
        else:
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        if data_args.streaming:
            metrics["eval_samples"] = max_eval_samples
        else:
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()




