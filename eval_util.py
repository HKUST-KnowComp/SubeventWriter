def get_latex_format_result(metric_list, result_dict):
    latex_format_result, score_list = "", []
    for metric in metric_list:
        if metric in ["bleu1", "bleu2"]:
            coarse_name = metric[:-1]
            score = result_dict[coarse_name]["precisions"][int(metric[-1]) - 1]
        elif metric in ["rouge1", "rouge2", "rougeL"]:
            coarse_name = metric[:-1]
            score = result_dict[coarse_name][metric]
        elif metric in ["meteor"]:
            score = result_dict[metric][metric]
        elif metric in ["bertscore"]:
            score = result_dict[metric]["f1"]
        elif metric in ["accuracy", "precision", "recall", "f1"]:
            score = result_dict[metric]
        else:
            raise ValueError("Wrong metric name")

        score *= 100
        score = round(score, 2)
        str_score = "{:.2f}".format(score)
        latex_format_result += "&" + str_score
        score_list.append(score)
    return latex_format_result, score_list


def get_latex_form_list(score_list):
    score_str = ""
    for score in score_list:
        score_str += "&" + str(score)
    return score_str
