import copy

import torch
import logging
import datasets
import numpy as np
from typing import NamedTuple, Tuple, Union, Optional
from transformers import Trainer
from datasets import Dataset
from transformers import Seq2SeqTrainer, TrainingArguments
from transformers.deepspeed import deepspeed_init
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify
from transformers.trainer_pt_utils import nested_truncate, IterableDatasetShard
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.file_utils import is_datasets_available
from torch.utils.data import DataLoader
from ds_train_util import decode_generation
from prompt_template.iterative_generation_template import iterative_generation_postprocess

logger = logging.getLogger()


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class ExtendedEvalLoopOutput(NamedTuple):
    input_ids: Union[np.ndarray, Tuple[np.ndarray]]
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    num_samples: Optional[int]


class ExtendedEvalBeamOutput(NamedTuple):
    input_ids: Union[np.ndarray, Tuple[np.ndarray]]
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    beam_scores: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    num_samples: Optional[int]


class ExtendedSeq2SeqTrainer(Seq2SeqTrainer):
    def get_generation_dataloader(self, test_dataset):
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(test_dataset, batch_size=self.args.eval_batch_size,
                                                    drop_last=False, num_processes=self.args.world_size,
                                                    process_index=self.args.process_index, )
            return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.data_collator,
                              num_workers=self.args.dataloader_num_workers,
                              pin_memory=self.args.dataloader_pin_memory, )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(test_dataset, sampler=test_sampler,
                          batch_size=self.args.eval_batch_size, collate_fn=self.data_collator, drop_last=False,
                          pin_memory=self.args.dataloader_pin_memory, )

    def generate(
            self,
            test_dataset,
            max_source_length=None,
            max_length=None,
            num_beams=None,
            min_length=None,
            do_sample=None,
            temperature=None,
            top_k=None):
        self._max_source_length = max_source_length
        self._max_length = max_length
        self._num_beams = num_beams
        self._min_length = min_length
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_k = top_k
        self._memory_tracker.start()

        #  self.callback_handler.on_evaluate(self.args, self.state, self.control)
        test_dataloader = self.get_generation_dataloader(test_dataset)
        with torch.no_grad():
            output = self.generation_loop(
                test_dataloader, description="Prediction")
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, {})
        return output

    def generation_loop(
            self,
            dataloader,
            description,
    ):
        args = self.args

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        input_ids_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_input_ids = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            input_ids, logits, labels = self.generation_step(inputs)

            # Update containers on host
            if input_ids is not None:
                input_ids = self._pad_across_processes(input_ids)
                input_ids = self._nested_gather(input_ids)
                input_ids_host = input_ids if input_ids_host is None else nested_concat(input_ids_host, input_ids,
                                                                                        padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if input_ids_host is not None:
                    input_ids = nested_numpify(input_ids_host)
                    all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids,
                                                                                          padding_index=-100)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                input_ids_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if input_ids_host is not None:
            input_ids = nested_numpify(input_ids_host)
            all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids,
                                                                                  padding_index=-100)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_input_ids is not None:
            all_input_ids = nested_truncate(all_input_ids, num_samples)
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        return ExtendedEvalLoopOutput(input_ids=all_input_ids, predictions=all_preds, label_ids=all_labels,
                                      num_samples=num_samples)

    def generation_step(self, inputs, ):
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "min_length": getattr(self, '_min_length', 4),
            "do_sample": getattr(self, "_do_sample", False),
            "temperature": getattr(self, "_temperature", 1.0),
            "top_k": getattr(self, "_top_k", 50),
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if self._max_source_length is None:
            self._max_source_length = self.model.config.max_length

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        if generation_inputs.shape[-1] < self._max_source_length:
            generation_inputs = self._pad_tensors_to_max_len(generation_inputs, self._max_source_length)

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        return generation_inputs, generated_tokens, labels

    def generate_by_self_talking(self,
                                 iter_generation_test_dataset,
                                 tokenizer,
                                 max_iterative_step,
                                 max_source_length=None,
                                 max_length=None,
                                 num_beams=None,
                                 min_length=None,
                                 do_sample=None,
                                 temperature=None,
                                 top_k=None):
        self.model.eval()
        # init the iterative dataset here
        epoch_step = 0
        while True:
            epoch_step += 1
            if epoch_step == max_iterative_step or len(iter_generation_test_dataset) == 0:
                break
            logger.info("generating the {}th step".format(epoch_step))
            test_results = self.generate(iter_generation_test_dataset,
                                         max_source_length,
                                         max_length,
                                         num_beams,
                                         min_length,
                                         do_sample,
                                         temperature,
                                         top_k)

            new_preds = decode_generation(test_results, tokenizer, iterative_generation_postprocess)[-2]
            iter_generation_test_dataset.update_unfinished_data(new_preds)
            iter_generation_test_dataset = copy.deepcopy(iter_generation_test_dataset)
            logger.info(
                "{} processes left after the {} step generated".format(len(iter_generation_test_dataset), epoch_step))

        total_data = iter_generation_test_dataset.total_data
        pred_list = iter_generation_test_dataset.total_preds
        input_list = [data["title"] for data in total_data]
        label_list = [data["subevents"] for data in total_data]

        for idx, pred in enumerate(pred_list):
            if "".join(pred) == "":  # change empty output
                pred_list[idx] = ["none"]

        text_list = [input_list, pred_list, label_list, input_list, pred_list, label_list]
        return text_list

    def generate_with_global_controller(self, iter_generation_test_dataset, tokenizer, max_iterative_step,
                                        coherence_model_path,
                                        max_source_length=None, max_length=None, num_beams=None,
                                        coherence_weight=1, min_length=None, do_sample=None, temperature=None,
                                        top_k=None):
        self.model.eval()
        # init the iterative dataset here
        epoch_step = 0
        while True:
            epoch_step += 1
            if epoch_step == max_iterative_step or len(iter_generation_test_dataset) == 0:
                break
            logger.info("generating the {}th step".format(epoch_step))
            test_results = self.beam_generate(iter_generation_test_dataset, max_source_length, max_length,
                                              num_beams, min_length, do_sample, temperature, top_k)

            new_inputs, new_preds, new_labels = decode_generation(test_results, tokenizer, iterative_generation_postprocess)[-3:]

            LM_scores = test_results.beam_scores.reshape(-1)
            # get coherence scores here
            coherence_score = self.get_coherence_score(
                coherence_model_path, iter_generation_test_dataset, num_beams,
                new_inputs, new_preds)
            total_scores = LM_scores + coherence_weight * coherence_score
            total_scores = total_scores.reshape(-1, num_beams)
            coherent_idx = total_scores.argmax(axis=1)

            coherent_preds = []
            for i, c_idx in zip(range(0, len(new_preds), num_beams), coherent_idx):
                if "none" in new_preds[i: i + num_beams]:  # == new_preds[i]:  # in new_preds[i: i + num_beams]:
                    coherent_preds.append("none")
                else:
                    coherent_preds.append(new_preds[i + c_idx])
            new_preds = coherent_preds

            iter_generation_test_dataset.update_unfinished_data(new_preds)
            iter_generation_test_dataset = copy.deepcopy(iter_generation_test_dataset)
            logger.info(
                "{} processes left after the {} step generated".format(len(iter_generation_test_dataset), epoch_step))

        total_data = iter_generation_test_dataset.total_data
        pred_list = iter_generation_test_dataset.total_preds
        input_list = [data["title"] for data in total_data]
        label_list = [data["subevents"] for data in total_data]

        for idx, pred in enumerate(pred_list):
            if "".join(pred) == "":  # change empty output
                pred_list[idx] = ["none"]

        text_list = [input_list, pred_list, label_list, input_list, pred_list, label_list]
        return text_list

    @staticmethod
    def get_coherence_score(coherence_model_path, test_dataset, num_beams,
                            new_inputs, new_preds, max_length=256, num_labels=2):
        config = AutoConfig.from_pretrained(coherence_model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(coherence_model_path, use_fast=True)
        coherence_model = AutoModelForSequenceClassification.from_pretrained(
            coherence_model_path,
            from_tf=bool(".ckpt" in coherence_model_path),
            config=config)

        prev_preds = [" ".join(pp) for pp in test_dataset.get_unfinished_data()["preds"]]
        prev_preds = [pp for pp in prev_preds for _ in range(num_beams)]
        new_inputs = [ipt for ipt in new_inputs for _ in range(num_beams)]
        full_preds = [prev_p + " " + cur_p for prev_p, cur_p in zip(prev_preds, new_preds)]
        result = tokenizer(new_inputs, full_preds, padding=False, max_length=max_length, truncation=True)
        coherence_dataset = Dataset.from_dict(result)
        # args = TrainingArguments(per_device_eval_batch_size=8, output_dir="./")
        coherence_trainer = Trainer(model=coherence_model,
                                    # args=args,
                                    train_dataset=None,
                                    eval_dataset=None,
                                    compute_metrics=None,
                                    tokenizer=tokenizer)
        coherence_score = coherence_trainer.predict(coherence_dataset).predictions
        coherence_score = torch.from_numpy(coherence_score)
        coherence_score = torch.nn.Softmax(dim=-1)(coherence_score)
        coherence_score = coherence_score[:, 1].cpu().detach().numpy()
        return coherence_score

    def beam_generate(self, test_dataset, max_source_length=None, max_length=None,
                      num_beams=None, min_length=None, do_sample=None, temperature=None, top_k=None):
        self._max_source_length = max_source_length
        self._max_length = max_length
        self._num_beams = num_beams
        self._min_length = min_length
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_k = top_k
        self._memory_tracker.start()

        #  self.callback_handler.on_evaluate(self.args, self.state, self.control)
        test_dataloader = self.get_generation_dataloader(test_dataset)
        with torch.no_grad():
            output = self.beam_generation_loop(
                test_dataloader, description="Prediction")
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, {})
        return output

    def beam_generation_loop(self, dataloader, description, ):
        args = self.args

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        input_ids_host, preds_host, labels_host, scores_host = None, None, None, None
        # losses/preds/labels on CPU (final containers)
        all_input_ids, all_preds, all_labels, all_scores = None, None, None, None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            input_ids, logits, scores, labels = self.beam_generation_step(inputs)

            # Update containers on host
            if input_ids is not None:
                input_ids = self._pad_across_processes(input_ids)
                input_ids = self._nested_gather(input_ids)
                input_ids_host = input_ids if input_ids_host is None else nested_concat(input_ids_host, input_ids,
                                                                                        padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if scores is not None:
                scores = self._pad_across_processes(scores)
                scores = self._nested_gather(scores)
                scores_host = scores if scores_host is None else nested_concat(scores_host, scores, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if input_ids_host is not None:
                    input_ids = nested_numpify(input_ids_host)
                    all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids,
                                                                                          padding_index=-100)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if scores_host is not None:
                    scores = nested_numpify(scores_host)
                    all_scores = (
                        scores if all_scores is None else nested_concat(all_scores, scores, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                input_ids_host, preds_host, labels_host, scores_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if input_ids_host is not None:
            input_ids = nested_numpify(input_ids_host)
            all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids,
                                                                                  padding_index=-100)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        if scores_host is not None:
            scores = nested_numpify(scores_host)
            all_scores = scores if all_scores is None else nested_concat(all_scores, scores, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        num_return_sequences = self._num_beams if self._num_beams is not None else self.model.config.num_beams
        total_num_samples = num_samples * num_return_sequences

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_input_ids is not None:
            all_input_ids = nested_truncate(all_input_ids, num_samples)
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, total_num_samples)
        if all_scores is not None:
            all_scores = nested_truncate(all_scores, total_num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        return ExtendedEvalBeamOutput(input_ids=all_input_ids, predictions=all_preds, beam_scores=all_scores,
                                      label_ids=all_labels, num_samples=num_samples)

    def beam_generation_step(self, inputs, ):
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "min_length": getattr(self, '_min_length', 4),
            "do_sample": getattr(self, "_do_sample", False),
            "temperature": getattr(self, "_temperature", 1.0),
            "top_k": getattr(self, "_top_k", 50),
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "num_return_sequences": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "return_dict_in_generate": True,
            "output_scores": True
        }

        if self._max_source_length is None:
            self._max_source_length = self.model.config.max_length

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_dict = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        generated_tokens = generated_dict.sequences
        generated_scores = generated_dict.sequences_scores.reshape(-1, 1)

        if generation_inputs.shape[-1] < self._max_source_length:
            generation_inputs = self._pad_tensors_to_max_len(generation_inputs, self._max_source_length)

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        return generation_inputs, generated_tokens, generated_scores, labels
