import random
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    set_seed,
)
from prompt_template import get_t5_postprocess, get_iterative_t5_postprocess
from ds_train_util import get_preprocess_function, init_optimizer
from ds_train_util import ds_init_logger, ds_init_output_dir, decode_and_store_generation
from ds_train_util import ModelArguments, DataTrainingArguments, ExtendedSeq2SeqTrainingArguments
from ds_train_util import is_main_process, format_args
from ds_train_util import IterativeGenerationDataset
from ds_seq2seqtrainer import ExtendedSeq2SeqTrainer
from prompt_template import get_iterative_generation_t5_template
from ds_train_util import store_generation, finish_checkpoint

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtendedSeq2SeqTrainingArguments))
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
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if not training_args.iterative:
        tokenizer.add_tokens(["[END_STEP]"])
    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        raise ValueError(
            f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
            f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings}."
        )

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
                data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang, tokenizer.tgt_lang = data_args.lang, data_args.lang
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    column_names = raw_datasets["train"].column_names
    preprocess_function = get_preprocess_function(data_args, column_names, tokenizer, iterative=training_args.iterative)
    if not training_args.iterative:
        postprocess_function = get_t5_postprocess(data_args.prompt_type)
    else:
        postprocess_function = get_iterative_t5_postprocess(data_args.prompt_type)

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_train_samples is not None:
        train_dataset = raw_datasets["train"]
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        raw_datasets["train"] = train_dataset

    with training_args.main_process_first(desc="raw datasets map pre-processing"):
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    valid_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"] if "test" in processed_datasets else None

    if is_main_process(training_args.local_rank):
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    optimizer = init_optimizer(training_args, model)

    # Initialize our Trainer
    trainer = ExtendedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Training
    logger.info("*** process {} starts training ***".format(training_args.local_rank))
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["local_rank"] = training_args.local_rank
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # set gradients to None to release GPU memory
        optimizer.zero_grad(set_to_none=True)
        # delete the optimizer to release GPU memory
        trainer.optimizer = None
        del optimizer

        if training_args.save_checkpoint:
            trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    gen_kwargs = {
        "max_source_length": data_args.max_source_length,
        "max_length": data_args.val_max_target_length,
        "num_beams": data_args.num_beams,
        "min_length": training_args.min_length,
        "do_sample": False,
        "temperature": 1.0,
    }
    if training_args.do_predict:
        logger.info("*** Predict process {} ***".format(training_args.local_rank))

        logger.info("local rank {} running on eval dataset".format(training_args.local_rank))
        eval_results = trainer.generate(valid_dataset, **gen_kwargs)
        if trainer.is_world_process_zero():
            decode_and_store_generation(training_args, eval_results, tokenizer, postprocess_function, "valid")

        if test_dataset is not None:
            logger.info("local rank {} running on test dataset".format(training_args.local_rank))
            test_results = trainer.generate(test_dataset, **gen_kwargs)
            if trainer.is_world_process_zero():
                decode_and_store_generation(training_args, test_results, tokenizer, postprocess_function, "test")

    # iterative generation here
    if training_args.do_iterative_predict:
        iterative_generation_template = get_iterative_generation_t5_template(data_args.prompt_type)
        iterative_generation_valid_dataset = IterativeGenerationDataset(
            data_args.validation_file, tokenizer,
            iterative_generation_template,
            data_args.max_source_length,
            data_args.val_max_target_length)

        if not training_args.global_controller:
            logger.info("local rank {} running on iterative eval dataset".format(training_args.local_rank))
            valid_text_list = trainer.generate_by_self_talking(iterative_generation_valid_dataset,
                                                               tokenizer, training_args.iterative_step, **gen_kwargs)
        else:
            logger.info("local rank {} running on iterative eval dataset".format(training_args.local_rank))
            valid_text_list = trainer.generate_with_global_controller(
                iterative_generation_valid_dataset,
                tokenizer, training_args.iterative_step,
                coherence_model_path=training_args.coherence_model_path,
                coherence_weight=training_args.coherence_weight,
                **gen_kwargs)
        if trainer.is_world_process_zero():
            store_generation(training_args, valid_text_list, "full_script_valid")

        if test_dataset is not None:
            iterative_generation_test_dataset = IterativeGenerationDataset(
                data_args.test_file, tokenizer,
                iterative_generation_template,
                data_args.max_source_length,
                data_args.val_max_target_length)
            if not training_args.global_controller:
                logger.info("local rank {} running on iterative test dataset".format(training_args.local_rank))
                test_text_list = trainer.generate_by_self_talking(iterative_generation_test_dataset,
                                                                  tokenizer, training_args.iterative_step, **gen_kwargs)
            else:
                logger.info("local rank {} running on iterative test dataset".format(training_args.local_rank))
                test_text_list = trainer.generate_with_global_controller(
                    iterative_generation_test_dataset,
                    tokenizer, training_args.iterative_step,
                    coherence_model_path=training_args.coherence_model_path,
                    coherence_weight=training_args.coherence_weight,
                    **gen_kwargs)
            if trainer.is_world_process_zero():
                store_generation(training_args, test_text_list, "full_script_test")

    if trainer.is_world_process_zero():
        finish_checkpoint(training_args, data_args, model_args)


if __name__ == "__main__":
    main()
