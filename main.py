import argparse
from config import *
from env import WordleEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wordle Agent using Reinforcement Learning")
    sub_parser = parser.add_subparsers(help="Choose run mode", dest="mode")

    train = sub_parser.add_parser("train")
    train.add_argument("-ep", type=int,
                       default=EPISODES, help="Number of episodes")
    train.add_argument("-lr", type=float,
                       default=LEARNING_RATE, help="Learning Rate")
    train.add_argument("-er", type=float,
                       default=EXPLORATION_RATE, help="Exploration Rate")
    train.add_argument("-df", type=float,
                       default=DISCOUNT_FACTOR, help="Discount Factor")

    sub_parser.add_parser("infer")
    sub_parser.add_parser("demo")

    args = parser.parse_args()


if __name__ == "__main__":
    main()
