import torch
from torch import nn
import numpy as np
from typing import Optional
from transformers import Trainer, EvalPrediction
from dataclasses import dataclass, field
from datasets import load_metric
from transformers import TrainingArguments


def get_metric_function():
    metric_dict = {"acc": load_metric("accuracy"), "p": load_metric("precision"),
                   "r": load_metric("recall"), "f1": load_metric("f1")}

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = {k: v
                  for metric_name, metric in metric_dict.items()
                  for k, v in metric.compute(predictions=preds, references=p.label_ids).items()}
        return result

    return compute_metrics


def get_classification_preprocess_function(data_args, tokenizer):
    sentence1_key, sentence2_key = data_args.sentence1_column, data_args.sentence2_column
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        sentence1_list = examples[sentence1_key]
        sentence2_list = [" ".join(subevent_seq) for subevent_seq in examples[sentence2_key]]

        result = tokenizer(sentence1_list, sentence2_list, padding=padding, max_length=max_seq_length, truncation=True)

        result["label"] = examples["label"]
        return result

    return preprocess_function


class CustomizedTrainer(Trainer):
    def update_negative_size(self, negative_size=4):
        self.my_negative_size = negative_size

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1 / self.my_negative_size, 1], device=labels.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    optimizer: str = field(default="AdamW", metadata={"help": "the optimizer to use, one of [Adafactor, AdamW]"})
    save_checkpoint: bool = field(default=False,
                                  metadata={"help": "whether to save the fine-tuned model"})
    negative_size: int = field(default=4,
                               metadata={"help": "the number of negative examples per positive one"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    sentence1_column: Optional[str] = field(default="title", metadata={
        "help": "the name of the column containing process titles"
    })
    sentence2_column: Optional[str] = field(default="subevents", metadata={
        "help": "the name of the column containing subevent sequences"
    })

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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
