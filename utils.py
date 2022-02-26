import json
import logging
import os
from time import time
from typing import List, Tuple

from config import LOG_DIRECTORY


def initialize_logger(filename: str = None, level: int = logging.DEBUG) -> None:
    if filename is None:
        filename = "{:.0f}".format(time())

    logging.basicConfig(
        filename=os.path.join(LOG_DIRECTORY, f"{filename}.txt"),
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def load_wordle_data() -> Tuple[List[str], List[str]]:
    allowed_guesses, answers = [], []

    with open(os.path.join("data", "allowed_guesses_without_answers.json"), "r") as f:
        allowed_guesses = json.load(f)
        f.close()

    with open(os.path.join("data", "answers.json"), "r") as f:
        answers = json.load(f)
        f.close()

    allowed_guesses = allowed_guesses + answers
    return allowed_guesses, answers
