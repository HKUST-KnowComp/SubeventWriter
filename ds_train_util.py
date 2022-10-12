import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import Seq2SeqTrainingArguments
import logging, os, shutil, datasets, transformers
from torch.utils.data import Dataset
from transformers import AdamW, Adafactor
from prompt_template import iterate_steps
from prompt_template import get_iterative_t5_template, get_t5_template


def finish_checkpoint(training_args, data_args, model_args):
    with open(os.path.join(training_args.output_dir, "checkpoint_finish"), "a") as fout:
        training_args_dict = {key: str(value) for key, value in training_args.__dict__.items()}
        data_args_dict = {key: str(value) for key, value in data_args.__dict__.items()}
        model_args_dict = {key: str(value) for key, value in model_args.__dict__.items()}
        total_args_dict = {**training_args_dict, **data_args_dict, **model_args_dict}
        fout.write(json.dumps(total_args_dict) + "\n")


def get_preprocess_function(args, column_names, tokenizer, iterative=False):
    text_column, summary_column = get_input_output_names(args, column_names)
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
    if iterative:
        prompt_function = get_iterative_t5_template(args.prompt_type)
    else:
        prompt_function = get_t5_template(args.prompt_type)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        if iterative:
            inputs, targets = iterate_steps(inputs, targets)
        inputs, targets = prompt_function(inputs, targets)
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def get_input_output_names(args, column_names):
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    return text_column, summary_column


def init_optimizer(args, model):
    if args.optimizer == "AdamW":
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.optimizer == "Adafactor":
        optimizer = Adafactor(params=model.parameters(), lr=args.learning_rate,
                              scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        raise ValueError("Wrong optimizer name")
    return optimizer


def is_main_process(local_rank):
    return local_rank == 0 or local_rank == -1


def decode_generation(predict_results, tokenizer, postprocess_function):
    input_ids = predict_results.input_ids
    preds = predict_results.predictions
    labels = predict_results.label_ids

    # Replace -100 as we can't decode them.
    input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    raw_inputs = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    raw_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    raw_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    text_list = postprocess_function(raw_inputs, raw_preds, raw_labels)
    return text_list


def store_generation(training_args, text_list, mode):
    if training_args.store_generation:
        with open(os.path.join(training_args.generation_path, "{}.json".format(mode)), "w") as fout:
            for ri, rp, rl, i, p, l in tqdm(zip(*text_list), "output generations"):
                fout.write(json.dumps({"input": i, "pred": p, "label": l,
                                       "raw_input": ri, "raw_pred": rp, "raw_label": rl}) + "\n")


def decode_and_store_generation(training_args, predict_results, tokenizer, postprocess_function, mode):
    text_list = decode_generation(predict_results, tokenizer, postprocess_function)
    store_generation(training_args, text_list, mode)


def ds_init_logger(training_args, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # init a formatter to add date information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # init a file handler and a stream handler
    fh = logging.FileHandler(os.path.join(training_args.output_dir, "train.log"), encoding="utf-8", mode="a")
    fh.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # set formatter to handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add those handlers to the root logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # the logger level of huggingface packages
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()

    return logger


def ds_init_output_dir(training_args):
    if training_args.do_train and os.path.exists(training_args.output_dir):
        if os.path.exists(os.path.join(training_args.output_dir, "checkpoint_finish")) > 0:
            raise ValueError(
                "training process in dir {} is finished, plz clear it manually.".format(training_args.output_dir))
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    os.system("touch {}".format(os.path.join(training_args.output_dir, "train.log")))
    if hasattr(training_args, "store_generation") and training_args.store_generation:
        training_args.generation_path = os.path.join(training_args.output_dir, "generation")
        if not os.path.exists(training_args.generation_path):
            os.makedirs(training_args.generation_path)


def format_args(args):
    args_as_dict = asdict(args)
    args_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in args_as_dict.items()}
    attrs_as_str = [f"{k}={v}," for k, v in sorted(args_as_dict.items())]
    return f"{args.__class__.__name__}\n({' '.join(attrs_as_str)})"


@dataclass
class ExtendedSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    min_length: int = field(default=3, metadata={"help": "minimum generation length"})
    iterative_step: int = field(default=10, metadata={"help": "the default number of steps in iterative generation"})
    store_generation: bool = field(default=True, metadata={"help": "store generation"})
    optimizer: str = field(default="AdamW", metadata={"help": "the optimizer to use, one of [Adafactor, AdamW]"})
    iterative: bool = field(default=False,
                            metadata={"help": "train the self-talk model if Yes. Otherwise, train all-at-once model"})
    save_checkpoint: bool = field(default=False,
                                  metadata={"help": "whether to save the fine-tuned model"})
    do_iterative_predict: bool = field(default=False,
                                       metadata={"help": "whether to evaluate the model iteratively"})
    global_controller: bool = field(default=False,
                                    metadata={"help": "whether use global controller to generate"})
    coherence_model_path: str = field(default=None,
                                      metadata={"help": "the path to the coherence model"})
    coherence_weight: float = field(default=1.0,
                                    metadata={"help": "weight of coherence model"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "be in one of [LM, sentinel]"}
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
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
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
                    "Useful for multilingual models like mBART where the first generated token"
                    "needs to be the target language token (Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.prompt_type is not None:
            assert self.prompt_type in ["LM", "sentinel"], "invalid prompt template type"
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


class IterativeGenerationDataset(Dataset):
    def __init__(self, file_path, tokenizer, preprocess_function,
                 max_source_length, max_target_length=10):
        with open(file_path) as fin:
            self.total_data = []
            for line in fin:
                self.total_data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.preprocess_function = preprocess_function
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.total_title = [data["title"] for data in self.total_data]
        self.total_subevents = [data["subevents"] for data in self.total_data]
        self.total_preds = [[] for _ in range(len(self.total_data))]
        self.unfinished_index = list(range(len(self.total_data)))
        self.unfinished_input = self._preprocess_function()

    def _preprocess_function(self):
        if not self.unfinished_index:
            return {}
        inputs = [self.total_title[idx] for idx in self.unfinished_index]
        targets = [self.total_subevents[idx] for idx in self.unfinished_index]
        predictions = [self.total_preds[idx] for idx in self.unfinished_index]

        assert len(set([len(i) for i in predictions])) <= 1, "length: {}".format(set([len(i) for i in predictions]))

        inputs, targets = self.preprocess_function(inputs, predictions, targets)
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=False, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=False, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __len__(self):
        return len(self.unfinished_index)

    def __getitem__(self, idx):
        return {"input_ids": self.unfinished_input["input_ids"][idx],
                "attention_mask": self.unfinished_input["attention_mask"][idx],
                "labels": self.unfinished_input["labels"][idx]}

    def update_unfinished_data(self, step_list):
        new_unfinished_index = []
        for step, idx in zip(step_list, self.unfinished_index):
            if step != "none":
                self.total_preds[idx].append(step)
                new_unfinished_index.append(idx)
        self.unfinished_index = new_unfinished_index
        self.unfinished_input = self._preprocess_function()

    def get_unfinished_data(self):
        inputs = [self.total_title[idx] for idx in self.unfinished_index]
        targets = [self.total_subevents[idx] for idx in self.unfinished_index]
        predictions = [self.total_preds[idx] for idx in self.unfinished_index]
        return {"inputs": inputs, "preds": predictions, "label": targets}
