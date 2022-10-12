# LM weak number template
def iterative_generation_LM_template(input_list, prediction_list, target_list):
    """
    :param input_list:
    :param target_list:
    :return: prompted_input: <title> Step 1: xxx Step 2: xxx Step3:
            prompted_target: xxx
    """
    if len(prediction_list) == 0:
        return [], []
    step = len(prediction_list[0]) + 1
    prompted_input_list, prompted_target_list = [], []
    for title, known_steps, target in zip(input_list, prediction_list, target_list):
        prompted_input = title
        if known_steps:
            prompted_input += " " + " ".join(["Step {}: {}".format(idx, s) for idx, s in enumerate(known_steps, start=1)])
        assert step == len(known_steps) + 1
        prompted_input += " Step {}:".format(step)
        prompted_target = target[min(step - 1, len(target) - 1)]  # index start from 0
        prompted_input_list.append(prompted_input)
        prompted_target_list.append(prompted_target)
    return prompted_input_list, prompted_target_list


def iterative_generation_postprocess(raw_input_ids, raw_preds, raw_labels):
    new_input_ids = [i.strip() for i in raw_input_ids]
    new_preds = [p.strip() for p in raw_preds]
    new_labels = [l.strip() for l in raw_labels]
    return raw_input_ids, raw_preds, raw_labels, new_input_ids, new_preds, new_labels


def get_t5_special_token(index):
    return "<extra_id_{}>".format(index)


# sentinel template
def iterative_generation_sentinel_template(input_list, prediction_list, target_list):
    """
    :param input_list:
    :param target_list:
    :return: prompted_input: <title> Step 1: xxx Step 2: xxx Step 3: <extra_id_0>
            prompted_target: <extra_id_0> xxx <extra_id_1>
    """
    if len(prediction_list) == 0:
        return [], []
    step = len(prediction_list[0]) + 1
    prompted_input_list, prompted_target_list = [], []
    for title, known_steps, target in zip(input_list, prediction_list, target_list):
        prompted_input = title
        if known_steps:
            prompted_input += " " + " ".join(["Step {}: {}".format(idx, s) for idx, s in enumerate(known_steps, start=1)])
        assert step == len(known_steps) + 1
        prompted_input += " Step {}: ".format(step) + get_t5_special_token(0)
        prompted_target = get_t5_special_token(0) + " " + target[min(step - 1, len(target) - 1)]
        prompted_target += " {}".format(get_t5_special_token(1))
        prompted_input_list.append(prompted_input)
        prompted_target_list.append(prompted_target)
    return prompted_input_list, prompted_target_list


# test iterate step
if __name__ == "__main__":
    title = "How to Get Out of Debt Quickly?"
    predictions = ["Hello", "World"]
    subevents = ["Prepare the area for cleaning.",
                 "Fill a small spray bottle with a mixture of vinegar and water.",
                 "Spray the window's surface.",
                 "Rub the window's entire surface with a cloth to work the vinegar in.",
                 "Dry the window's entire surface."]

    for i in range(len(predictions) + 1):
        prompted_input, prompted_target = iterative_generation_LM_template([title], [predictions[:i]], [subevents])
        print(prompted_input, prompted_target)
        prompted_input, prompted_target = iterative_generation_sentinel_template([title], [predictions[:i]], [subevents])
        print(prompted_input, prompted_target)