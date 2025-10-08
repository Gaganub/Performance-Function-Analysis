import argparse


def create_argument_parser():
    """
    Construct and return the argument parser for performance function experiments.
    """
    parser = argparse.ArgumentParser(
        prog="PerformanceExperimentRunner",
        description="Execute experiments on performance function models."
    )

    # General configuration
    parser.add_argument(
        "--lang", "-l",
        metavar="LANG",
        type=str,
        default="all",
        help="Language to use for experiments. Use 'all' to include every supported language."
    )

    parser.add_argument(
        "--pivot-size", "-p",
        default="all",
        help="Specify pivot size for experiments. Use 'all' to include all supported sizes."
    )

    parser.add_argument(
        "--c12",
        type=float,
        default=0.1,
        help="Cost ratio between unit translation and unit manual data."
    )

    # Experiment mode and seed
    parser.add_argument(
        "--mode", "-m",
        nargs="+",
        default=["fit_nd_eval"],
        choices=["fit_nd_eval", "expansion_paths"],
        help="Select experiment mode(s)."
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed value for reproducibility."
    )

    # File and directory configuration
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="performance_data/",
        help="Path to the directory containing performance data."
    )

    parser.add_argument(
        "--performance-file", "-f",
        type=str,
        default="tydiqa_mbert_results.csv",
        help="Name of the CSV file that holds performance data."
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/",
        help="Directory to save experiment outputs."
    )

    def add_common_arguments(parser):
    """Attach commonly used experiment arguments to the parser."""
    parser.add_argument(
        "--test_split_fraction",
        type=float,
        default=0.2,
        help="Fraction of the dataset to be used as the test split."
    )
    return parser


def get_args():
    """Initialize the argument parser, add arguments, and return parsed arguments."""
    parser = create_argument_parser()  # assumes base parser function exists
    parser = add_common_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = get_args()
    print(vars(arguments))  # display arguments as a dictionary
