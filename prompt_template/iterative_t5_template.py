import copy


def iterate_steps(title_list, subevents_list):
    input_list, target_list = [], []
    for title, subevents in zip(title_list, subevents_list):
        subevents = copy.deepcopy(subevents) + ["none"]
        for i in range(len(subevents)):
            input = [title, subevents[:i]]
            target = subevents[i]
            input_list.append(input)
            target_list.append(target)
    return input_list, target_list


# LM weak number template
def iterative_LM_template(input_list, target_list):
    """
    :param input_list:
    :param target_list:
    :return: prompted_input: <title> Step 1: xxx Step 2: xxx Step3:
            prompted_target: xxx
    """
    prompted_input_list, prompted_target_list = [], []
    for input, target in zip(input_list, target_list):
        title, known_steps = input
        prompted_input = title
        if known_steps:
            prompted_input += " " + " ".join(["Step {}: {}".format(idx, s) for idx, s in enumerate(known_steps, start=1)])
        prompted_input += " Step {}:".format(len(known_steps) + 1)
        prompted_target = target
        prompted_input_list.append(prompted_input)
        prompted_target_list.append(prompted_target)
    return prompted_input_list, prompted_target_list


def iterative_postprocess(input_ids, preds, labels):
    raw_input_ids, raw_preds, raw_labels = [], [], []
    new_input_ids, new_preds, new_labels = [], [], []
    for in_id, pred, label in zip(input_ids, preds, labels):
        new_input_ids.append(in_id.strip())
        new_preds.append([pred.strip()])
        new_labels.append([label.strip()])

        raw_input_ids.append(in_id)
        raw_preds.append([pred])
        raw_labels.append([label])
    return raw_input_ids, raw_preds, raw_labels, new_input_ids, new_preds, new_labels


def get_t5_special_token(index):
    return "<extra_id_{}>".format(index)


# sentinel template
def iterative_sentinel_template(input_list, target_list):
    """
    :param input_list:
    :param target_list:
    :return: prompted_input: <title> Step 1: xxx Step 2: xxx Step 3: <extra_id_0>
            prompted_target: <extra_id_0> xxx <extra_id_1>
    """
    prompted_input_list, prompted_target_list = [], []
    for input, target in zip(input_list, target_list):
        title, known_steps = input
        prompted_input = title
        if known_steps:
            prompted_input += " " + " ".join(["Step {}: {}".format(idx, s) for idx, s in enumerate(known_steps, start=1)])
        prompted_input += " Step {}: ".format(len(known_steps) + 1) + get_t5_special_token(0)
        prompted_target = get_t5_special_token(0) + " " + target
        prompted_target += " {}".format(get_t5_special_token(1))
        prompted_input_list.append(prompted_input)
        prompted_target_list.append(prompted_target)
    return prompted_input_list, prompted_target_list


# test here
# if __name__ == "__main__":
#     title = ["How to Get Out of Debt Quickly?",
#              ["Prepare the area for cleaning.",
#                  "Fill a small spray bottle with a mixture of vinegar and water.",
#                  "Spray the window's surface.",
#                  "Rub the window's entire surface with a cloth to work the vinegar in."]]
#     subevents = "Dry the window's entire surface."
#     input_list, target_list = iterative_LM_template([title], [subevents])
#     output = iterative_LM_postprocess(input_list, target_list, target_list)
#     input_list, target_list = iterative_sentinel_template([title], [subevents])
#     output = iterative_sentinel_postprocess(input_list, target_list, target_list)

# test iterate step
# if __name__ == "__main__":
#     title = "How to Get Out of Debt Quickly?"
#     subevents = ["Prepare the area for cleaning.",
#                  "Fill a small spray bottle with a mixture of vinegar and water.",
#                  "Spray the window's surface.",
#                  "Rub the window's entire surface with a cloth to work the vinegar in.",
#                  "Dry the window's entire surface."]
#
#     times = 120000
#     title_list = [title for _ in range(times)]
#     subevents_list = [subevents for _ in range(times)]
#     iterate_steps(title_list, subevents_list)
