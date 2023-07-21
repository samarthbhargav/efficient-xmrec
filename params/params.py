import argparse
import json
import os
import copy
from collections import OrderedDict
from typing import List, Dict

import numpy as np


def prod_dict(hyp_dict) -> List[Dict]:
    hyperparameters = OrderedDict(
        sorted(hyp_dict.items(), key=lambda _: _[0]))
    keys = list(hyperparameters.keys())
    indices = [len(values) for (arg, values) in hyperparameters.items()]
    choices = []
    for idx_choice in np.ndindex(*indices):
        one_choice = {}
        for arg_idx, (arg, val_idx) in enumerate(zip(keys, idx_choice)):
            one_choice[arg] = hyperparameters[arg][val_idx]
        choices.append(one_choice)

    return choices


def make_cmd(params: Dict) -> str:
    cmd = ""
    for p, v in params.items():
        if isinstance(v, bool):
            # only add if it's True
            if v:
                cmd += f"--{p}  "
        else:
            cmd += f"--{p} {v} "

    return " " + cmd
