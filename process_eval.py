import os
import json
import logging
import argparse
from tqdm import tqdm
from eval_util import get_latex_format_result
from collections import defaultdict
from eval_function.unify_metrics_api import AutoScorer


def change_process_form(process, input_column, pred_column, label_column):
    input_list = []
    pred_list = []
    label_list = []
    for p in process:
        input_list.append(p[input_column])
        pred_list.append(p[pred_column])
        label_list.append(p[label_column])
    return input_list, pred_list, label_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, default="/home/zwanggy/large_files/event_outputs/top1_jaccard")
    parser.add_argument("--input_column", type=str, default="input")
    parser.add_argument("--pred_column", type=str, default="pred")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--metric_list", type=str,
                        default="bleu1,bleu2,rougeL,bertscore")
    parser.add_argument("--result_list", action="store_true")
    parser.add_argument("--bert_score_model", type=str, help="the model path to bert score",
                        default="microsoft/deberta-xlarge-mnli")
    args = parser.parse_args()
    args.metric_list = [s.strip() for s in args.metric_list.split(",")]

    log_level = 20
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # init a formatter to add date information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # init a file handler and a stream handler
    fh = logging.FileHandler(os.path.join(args.gen_path, "performance_test.txt"), encoding="utf-8", mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    # add those handlers to the root logger
    logger.addHandler(fh)

    logger.info(args)

    result_list = []
    if args.result_list:
        for file in os.listdir(args.gen_path):
            full_path = os.path.join(args.gen_path, file)
            if not os.path.isdir(full_path):
                continue
            if not os.path.exists(os.path.join(full_path, "generation")):
                continue
            result_list.append(full_path)
    else:
        result_list = [args.gen_path]

    # init scorer
    metric_set = {"bleu", "rouge", "bertscore"}
    metric_kwargs = {"bleu": {"max_order": 2}, "rouge": {"use_stemmer": True},
                     "bertscore": {"batch_size": 32, "model_type": args.bert_score_model}}
    auto_scorer = AutoScorer(metric_set, reload=False)
    print("finish metric loading")

    grouped_performance = defaultdict(list)

    # init process
    for res_dir in result_list:
        eval_file_list = []
        generation_dir = os.path.join(res_dir, "generation/")
        for eval_file in os.listdir(generation_dir):
            if eval_file.endswith(".json") and ("valid" in eval_file or "test" in eval_file):
                eval_file_list.append(os.path.join(generation_dir, eval_file))

        for eval_file_path in eval_file_list:
            print(eval_file_path)
            with open(eval_file_path) as fin:
                process = []
                for line in tqdm(fin, "loading {}".format(eval_file_path.split("/")[-1])):
                    process.append(json.loads(line))

            input_list, pred_list, label_list = change_process_form(process, args.input_column,
                                                                    args.pred_column, args.label_column)
            result_dict = auto_scorer.compute(inputs=input_list, preds=pred_list,
                                              labels=label_list, metric_kwargs=metric_kwargs)

            latex_format_result, score_list = get_latex_format_result(args.metric_list, result_dict)
            file_name = eval_file_path.split("/")[-1]
            grouped_performance[file_name].append((eval_file_path, latex_format_result, score_list))
            print(latex_format_result)
            logger.info(eval_file_path)
            logger.info(str(result_dict))
            logger.info(latex_format_result)
            logger.info("------------------split line------------------")

    logger.info("------------------grouped result------------------")
    for key in grouped_performance:
        logger.info("------------------grouped key: {}------------------".format(key))
        for path, str_score, score_list in sorted(grouped_performance[key], key=lambda x: x[0]):
            logger.info(path)
            logger.info(str_score)

    logger.info("------------------best result------------------")
    main_key = "full_script_valid.json" if "full_script_valid.json" in grouped_performance else "valid.json"
    best_sum, best_path = 0, None
    for path, _, score_list in sorted(grouped_performance[main_key], key=lambda x: x[0]):
        if sum(score_list) > best_sum:
            best_sum = sum(score_list)
            best_path = path
    best_dir = "/".join(best_path.split("/")[:-1])
    for key in grouped_performance:
        logger.info("------------------best of key: {}------------------".format(key))
        for path, str_score, score_list in sorted(grouped_performance[key], key=lambda x: x[0]):
            cur_dir = "/".join(path.split("/")[:-1])
            if cur_dir == best_dir:
                logger.info("best path: {}".format(path))
                logger.info("best score: {}".format(str_score))
                logger.info("")
