# !/usr/bin/env python
import random

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from train_coherence_util import DataTrainingArguments, ModelArguments
from ds_train_util import ds_init_logger, ds_init_output_dir
from ds_train_util import is_main_process, format_args
from ds_train_util import finish_checkpoint, init_optimizer
from train_coherence_util import get_classification_preprocess_function
from train_coherence_util import get_metric_function, ExtendedTrainingArguments
from train_coherence_util import CustomizedTrainer


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtendedTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()

    if is_main_process(training_args.local_rank):
        ds_init_output_dir(training_args)

    with training_args.main_process_first(desc="getting logger"):
        logger = ds_init_logger(training_args, log_level)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}, " +
        f"bf16 training: {training_args.bf16}"
    )

    if is_main_process(training_args.local_rank):
        logger.info(format_args(training_args))
        logger.info(format_args(data_args))
        logger.info(format_args(model_args))

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    # Labels
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    # Preprocessing the raw_datasets

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    column_names = raw_datasets["train"].column_names
    preprocess_function = get_classification_preprocess_function(data_args, tokenizer)
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]

    # Log a few random samples from the training set:
    if is_main_process(training_args.local_rank):
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    compute_metrics = get_metric_function()
    optimizer = init_optimizer(training_args, model)
    # Initialize our Trainer
    trainer = CustomizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        data_collator=data_collator,
    )
    trainer.update_negative_size(training_args.negative_size)

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if training_args.save_checkpoint:
            trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        for name, eval_dataset in zip(["valid", "test"], [valid_dataset, test_dataset]):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)
            metrics = {key.replace("eval", name): score for key, score in metrics.items()}

            trainer.log_metrics(name, metrics)
            trainer.save_metrics(name, metrics)

    if trainer.is_world_process_zero():
        finish_checkpoint(training_args, data_args, model_args)


if __name__ == "__main__":
    main()
