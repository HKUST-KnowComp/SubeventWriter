from .t5_template import LM_template, sentinel_template
from .t5_template import LM_postprocess, sentinel_postprocess
from .iterative_t5_template import iterative_LM_template, iterative_sentinel_template
from .iterative_t5_template import iterate_steps, iterative_postprocess
from .iterative_generation_template import iterative_generation_sentinel_template
from .iterative_generation_template import iterative_generation_LM_template
from .iterative_generation_template import iterative_generation_postprocess


def get_iterative_generation_t5_template(prompt_type):
    if prompt_type == "LM":
        return iterative_generation_LM_template
    elif prompt_type == "sentinel":
        return iterative_generation_sentinel_template
    else:
        raise ValueError("Wrong prompt template type: {}".format(prompt_type))


def get_iterative_generation_t5_postprocess(prompt_type):
    if prompt_type == "LM":
        return iterative_generation_postprocess
    elif prompt_type == "sentinel":
        return iterative_generation_postprocess
    else:
        raise ValueError("Wrong prompt template type: {}".format(prompt_type))


def get_t5_template(prompt_type):
    if prompt_type == "LM":
        return LM_template
    elif prompt_type == "sentinel":
        return sentinel_template
    else:
        raise ValueError("Wrong prompt template type: {}".format(prompt_type))


def get_t5_postprocess(prompt_type):
    if prompt_type == "LM":
        return LM_postprocess
    elif prompt_type == "sentinel":
        return sentinel_postprocess
    else:
        raise KeyError("Wrong prompt template type: {}".format(prompt_type))


def get_iterative_t5_template(prompt_type):
    if prompt_type == "LM":
        return iterative_LM_template
    elif prompt_type == "sentinel":
        return iterative_sentinel_template
    else:
        raise ValueError("Wrong iterative prompt template type: {}".format(prompt_type))


def get_iterative_t5_postprocess(prompt_type):
    if prompt_type == "LM":
        return iterative_postprocess
    elif prompt_type == "sentinel":
        return iterative_postprocess
    else:
        raise KeyError("Wrong iterative prompt template type: {}".format(prompt_type))