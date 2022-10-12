import os
from datasets import load_metric
from collections import defaultdict


class AutoScorer:
    def __init__(self, metric_names, reload=True):
        self.rouge = None
        self.bleu = None
        self.bertscore = None
        self.reload = reload

        self._load_metric(metric_names)

    def _load_metric(self, metric_names):
        metric_names = set(metric_names)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = dir_path
        if "rouge" in metric_names:
            self.rouge = load_metric(os.path.join(dir_path, "my_rouge.py"))
            metric_names.remove("rouge")
        if "bleu" in metric_names:
            self.bleu = load_metric(os.path.join(dir_path, "my_bleu.py"))
            metric_names.remove("bleu")
        if "bertscore" in metric_names:
            if self.reload:
                self.bertscore = "unloaded_metric"
            else:
                self.bertscore = load_metric(os.path.join(self.dir_path, "bertscore.py"))
            metric_names.remove("bertscore")

        assert len(metric_names) == 0, "there are not found metric names: {}".format(metric_names)

    def compute(self, inputs, preds, labels, metric_kwargs):
        result = {}

        inputs = [" ".join(event_seq) for event_seq in inputs]

        input2labels = defaultdict(set)

        # group label here
        for i, l in zip(inputs, labels):
            input2labels[i].add(l)

        # I guess there are not multi-references problem
        for i, l in input2labels.items():
            assert len(l) == 1

        labels = []
        for i in inputs:
            labels.append(list(input2labels[i]))

        if self.rouge is not None:
            result["rouge"] = self.rouge.compute(predictions=preds, references=labels, **metric_kwargs["rouge"])
        if self.bleu is not None:
            result["bleu"] = self.bleu.compute(predictions=preds, references=labels, **metric_kwargs["bleu"])
        if self.bertscore is not None:
            if self.reload:
                self.bertscore = load_metric(os.path.join(self.dir_path, "bertscore.py"))
                result["bertscore"] = self.bertscore.compute(predictions=preds, references=labels, **metric_kwargs["bertscore"])
                self.bertscore = "unloaded_metric"
            else:
                result["bertscore"] = self.bertscore.compute(predictions=preds, references=labels,
                                                             **metric_kwargs["bertscore"])

        return result