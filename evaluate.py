import argparse
from utils.search import Text2Img
from utils.utils import read_txt


def get_cli_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_file", help="Path to labels file", type=str)

    return parser.parse_args()


if __name__ == '__main__':

    # Get CLI arguments and parse them
    args = get_cli_arg()
    labels_filepath = args.labels_file

    # Instantiate text 2 image system
    text2img = Text2Img()

    # Read test dataset
    test_dataset = read_txt(path=labels_filepath)

    # Perform evaluation
    accuracy, mapping = text2img.avg_precision_at_k(test_dataset, k=30)
    print(f"For the queries provided, the accuracy of the system is {accuracy}.")
