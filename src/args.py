import argparse

def build_parser():
    parser = argparse.ArgumentParser(description="Run Performance Function Experiments")

    parser.add_argument(
        "-l", "--lang",
        type=str,
        default="all",
        metavar="LANG",
        help="Specify the language for experiments. Use 'all' to run across all supported languages."
    )

    parser.add_argument(
        "-p", "--pivot_size",
        default="all",
        help="Pivot size for experiments. Use 'all' to include all supported sizes."
    )

    parser.add_argument(
        "--c12",
        type=float,
        default=0.1,
        help="Ratio between unit translation and unit manual data cost."
    )

    parser.add_argument(
        "-m", "--mode",
        nargs="+",
        default=["fit_nd_eval"],
        help="Experiment mode(s): 'fit_nd_eval' for fitting, 'expansion_paths' for path generation."
    )

    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default="performance_data/",
        help="Directory containing performance data."
    )

    parser.add_argument(
        "-f", "--performance_file",
        type=str,
        default="tydiqa_mbert_results.csv",
        help="CSV file containing performance data."
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="outputs/",
        help="Directory to store outputs."
    )

    parser.add_argument(
        "--test_split_frac",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing."
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    return parser.parse_args()
