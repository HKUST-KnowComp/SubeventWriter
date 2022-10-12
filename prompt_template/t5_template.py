import re


# LM weak number template
def LM_template(title_list, subevents_list):
    """
    :param title_list:
    :param subevents_list:
    :return: prompted_input: <title>
            prompted_target: Step 1: xxx [END_STEP] Step 2: xxx [END_STEP] ...
    """
    prompted_input_list, prompted_target_list = [], []
    for title, subevents in zip(title_list, subevents_list):
        prompted_input = title  # + " " + "{} steps".format(len(subevents))
        prompted_target = " ".join(["Step {}: {} [END_STEP]".format(idx, sub) for idx, sub in enumerate(subevents, start=1)])
        prompted_input_list.append(prompted_input)
        prompted_target_list.append(prompted_target)
    return prompted_input_list, prompted_target_list

STEP = re.compile(r"^Step \d:?")


def LM_postprocess(raw_input_ids, raw_preds, raw_labels):
    input_ids = [in_id.strip() for in_id in raw_input_ids]
    preds = [pred.strip() for pred in raw_preds]
    labels = [label.strip() for label in raw_labels]

    preds = [[STEP.sub("", step.strip()) for step in pred.split("[END_STEP]")] for pred in preds]
    labels = [[STEP.sub("", step.strip()) for step in label.split("[END_STEP]")] for label in labels]

    preds = [[step.strip() for step in pred if step.strip()] for pred in preds]
    labels = [[step.strip() for step in label if step.strip()] for label in labels]

    preds = [pred if pred else [""] for pred in preds]
    labels = [label if label else [""] for label in labels]

    return raw_input_ids, raw_preds, raw_labels, input_ids, preds, labels


def get_t5_special_token(index):
    return "<extra_id_{}>".format(index)


# sentinel template
def sentinel_template(title_list, subevents_list):
    """
    :param title_list:
    :param subevents_list:
    :return: prompted_input: <title> <extra_id_0>
            prompted_target: <extra_id_0> xxx [END_STEP] xxx [END_STEP] xxx [END_STEP] <extra_id_1>
    """
    prompted_input_list, prompted_target_list = [], []
    for title, subevents in zip(title_list, subevents_list):
        prompted_input = title + " " + get_t5_special_token(0)
        prompted_target = get_t5_special_token(0) + " " + " ".join(["{} [END_STEP]".format(sub) for sub in subevents])
        prompted_target += " {}".format(get_t5_special_token(1))
        prompted_input_list.append(prompted_input)
        prompted_target_list.append(prompted_target)
    return prompted_input_list, prompted_target_list


def sentinel_postprocess(raw_input_ids, raw_preds, raw_labels):
    input_ids = [in_id.strip() for in_id in raw_input_ids]
    preds = [pred.strip() for pred in raw_preds]
    labels = [label.strip() for label in raw_labels]

    preds = [pred.split("[END_STEP]") for pred in preds]
    labels = [label.split("[END_STEP]") for label in labels]

    preds = [[step.strip() for step in pred if step.strip()] for pred in preds]
    labels = [[step.strip() for step in label if step.strip()] for label in labels]

    preds = [pred if pred else [""] for pred in preds]
    labels = [label if label else [""] for label in labels]

    return raw_input_ids, raw_preds, raw_labels, input_ids, preds, labels



# test here
# if __name__ == "__main__":
#     title = "How to Get Out of Debt Quickly?"
#     subevents = ["Prepare the area for cleaning.",
#                  "Fill a small spray bottle with a mixture of vinegar and water.",
#                  "Spray the window's surface.",
#                  "Rub the window's entire surface with a cloth to work the vinegar in.",
#                  "Dry the window's entire surface."]
#     input_list, target_list = LM_template([title], [subevents])
#     input_list, target_list = sentinel_template([title], [subevents])
