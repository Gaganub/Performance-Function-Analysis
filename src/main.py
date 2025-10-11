import os
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Dict, Tuple, List, Optional, Any, NamedTuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(funcName)s() :: %(message)s"
)
logger = logging.getLogger("PFA")
logger.setLevel(logging.INFO)


@dataclass
class PerformanceFunctionConfig:
    """Configuration class for the performance function analyzer."""
    data_dir: str
    performance_file: str
    output_dir: str
    lang: str
    pivot_size: str
    mode: List[str]
    test_split_frac: float
    seed: int
    c12: float


class ModelEvaluationResults(NamedTuple):
    """Data structure for holding model evaluation results."""
    pred_error_df: pd.DataFrame
    params: Dict[str, Any]
    mae: float
    rmse: float
    r2: float


class PerformanceFunctionAnalyzer:
    """A class to analyze performance functions by fitting and evaluating models."""
    # Constants for column names and keys
    TGT_LANG_COL = "tgt_lang"
    PIVOT_SIZE_COL = "en_pivot_size"
    F1_SCORE_COL = "F1-Score"
    PREDICTED_F1_COL = "Predicted F1-Score"
    ABS_ERROR_COL = "Absolute Errors"
    SQ_ERROR_COL = "Squared Errors"
    
    # Constants for directories
    FIT_RESULTS_DIR = "fit_results"
    EXP_PATHS_DIR = "exp_paths"

    def __init__(self, config: PerformanceFunctionConfig):
        self.config = config
        self.perf_df: Optional[pd.DataFrame] = None
        self.langs: List[str] = []
        self.pivot_sizes: List[int] = []

    def load_data(self) -> None:
        """Loads and filters the performance data from a CSV file."""
        perf_filepath = os.path.join(self.config.data_dir, self.config.performance_file)
        logger.info(f"Loading data from '{perf_filepath}'...")
        self.perf_df = pd.read_csv(perf_filepath)

        self.langs = sorted(list(self.perf_df[self.TGT_LANG_COL].unique())) if self.config.lang == "all" else [self.config.lang]
        self.pivot_sizes = sorted(list(self.perf_df[self.PIVOT_SIZE_COL].unique())) if self.config.pivot_size == "all" else [int(self.config.pivot_size)]

        self.perf_df = self.perf_df[
            (self.perf_df[self.TGT_LANG_COL].isin(self.langs)) &
            (self.perf_df[self.PIVOT_SIZE_COL].isin(self.pivot_sizes))
        ]
        logger.info(f"Data loaded successfully. Shape: {self.perf_df.shape}")

    def create_output_dirs(self) -> None:
        """Creates the necessary output directories for saving results."""
        logger.info("Creating output directories if they don't exist...")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if "fit_nd_eval" in self.config.mode:
            os.makedirs(os.path.join(self.config.output_dir, self.FIT_RESULTS_DIR), exist_ok=True)
            
        if "expansion_paths" in self.config.mode:
            os.makedirs(os.path.join(self.config.output_dir, self.EXP_PATHS_DIR), exist_ok=True)

    def fit_and_evaluate(self, model_type: str, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> ModelEvaluationResults:
        """Fits and evaluates a performance function model."""
        pred_error_dfs = []
        lang2ps2parms = {}

        for lang, lang_sub_train_df in train_df.groupby(self.TGT_LANG_COL):
            if lang == "en":
                continue
            
            lang2ps2parms[lang] = {}
            for pivot_size, lang_ps_sub_train_df in lang_sub_train_df.groupby(self.PIVOT_SIZE_COL):
                logger.info(f"Processing Lang: {lang}, Pivot Size: {pivot_size}, Train Size: {len(lang_ps_sub_train_df)}")
                
                try:
                    if test_df is not None:
                        lang_ps_sub_test_df = test_df[
                            (test_df[self.TGT_LANG_COL] == lang) &
                            (test_df[self.PIVOT_SIZE_COL] == pivot_size)
                        ]
                        if model_type == "amue":
                            from src.fit_performance_functions import fit_nd_eval_amue_model_diff_test
                            params, pred_nd_error_df = fit_nd_eval_amue_model_diff_test(lang_ps_sub_train_df, lang_ps_sub_test_df, lang, pivot_size)
                        else:
                            from src.fit_performance_functions import fit_nd_eval_gpr_model_diff_test
                            params, pred_nd_error_df = fit_nd_eval_gpr_model_diff_test(lang_ps_sub_train_df, lang_ps_sub_test_df, lang, pivot_size)
                    else:
                        if model_type == "amue":
                            from src.fit_performance_functions import fit_nd_eval_amue_model
                            params, pred_nd_error_df = fit_nd_eval_amue_model(lang_ps_sub_train_df, lang, pivot_size)
                        else:
                            from src.fit_performance_functions import fit_nd_eval_gpr_model
                            params, pred_nd_error_df = fit_nd_eval_gpr_model(lang_ps_sub_train_df, lang, pivot_size)
                            
                except ValueError as e:
                    logger.warning(f"Skipping {lang}-{pivot_size} due to ValueError: {e}")
                    continue
                
                pred_error_dfs.append(pred_nd_error_df)
                lang2ps2parms[lang][pivot_size] = params.tolist() if isinstance(params, np.ndarray) else params

        if not pred_error_dfs:
            logger.error("No data was processed. Returning empty results.")
            return ModelEvaluationResults(pd.DataFrame(), {}, 0.0, 0.0, 0.0)

        pred_error_df = pd.concat(pred_error_dfs, axis=0)
        mae = pred_error_df[self.ABS_ERROR_COL].mean()
        rmse = np.sqrt(pred_error_df[self.SQ_ERROR_COL].mean())
        r2 = r2_score(pred_error_df[self.F1_SCORE_COL].values, pred_error_df[self.PREDICTED_F1_COL].values)

        return ModelEvaluationResults(pred_error_df, lang2ps2parms, mae, rmse, r2)

    def run_fit_and_eval(self) -> None:
        """Runs the full fitting and evaluation process for configured models."""
        if "fit_nd_eval" not in self.config.mode:
            return
            
        logger.info(f"Splitting performance data with test fraction {self.config.test_split_frac}...")
        perf_train_df, perf_test_df = train_test_split(
            self.perf_df, test_size=self.config.test_split_frac, random_state=self.config.seed
        )

        logger.info("Fitting and Evaluating AMUE Performance Function...")
        amue_results = self.fit_and_evaluate("amue", perf_train_df, perf_test_df)
        logger.info(f"AMUE Evaluation Complete | MAE: {amue_results.mae:.4f} | RMSE: {amue_results.rmse:.4f} | R²: {amue_results.r2:.4f}")

        pred_dir = os.path.join(self.config.output_dir, self.FIT_RESULTS_DIR)
        
        logger.info("Saving AMUE predictions and errors...")
        pred_file = os.path.join(pred_dir, f"amue_pred_nd_errors_lang{self.config.lang}_pivotSize{self.config.pivot_size}.csv")
        amue_results.pred_error_df.to_csv(pred_file, index=False)

        logger.info("Saving AMUE parameters...")
        params_file = os.path.join(pred_dir, f"amue_params_lang{self.config.lang}_pivotSize{self.config.pivot_size}.json")
        try:
            with open(params_file, "w") as f:
                json.dump(amue_results.params, f, indent=4, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save parameters to {params_file}: {e}")

        logger.info("Fitting and Evaluating GPR Performance Function...")
        gpr_results = self.fit_and_evaluate("gpr", perf_train_df, perf_test_df)
        logger.info(f"GPR Evaluation Complete | MAE: {gpr_results.mae:.4f} | RMSE: {gpr_results.rmse:.4f} | R²: {gpr_results.r2:.4f}")

    def analyze_expansion_paths(self) -> None:
        """Analyzes and plots the expansion paths based on the fitted AMUE model."""
        if "expansion_paths" not in self.config.mode:
            return
            
        logger.info("Fitting AMUE Performance Function on the entire dataset for expansion path analysis...")
        amue_results = self.fit_and_evaluate("amue", self.perf_df)
        amue_lang2ps2parms = amue_results.params
        
        from src.perf_func_analysis import get_expansion_path
        
        for lang in self.langs:
            if lang == "en":
                continue
            for pivot_size in self.pivot_sizes:
                if pivot_size == 0:
                    continue
                
                params = amue_lang2ps2parms.get(lang, {}).get(pivot_size)
                if not params:
                    logger.warning(f"No parameters found for lang={lang}, pivot_size={pivot_size}. Skipping.")
                    continue

                a0 = params[0]
                max_y = self.perf_df[
                    (self.perf_df[self.TGT_LANG_COL] == lang) & 
                    (self.perf_df[self.PIVOT_SIZE_COL] == pivot_size)
                ]["f1_score"].max()
                
                ys = np.linspace(np.ceil(a0), max_y, int((max_y - a0) // 1))
                
                get_expansion_path(
                    params, ys, self.config.c12, lang, pivot_size,
                    plot=True,
                    save_dir=os.path.join(self.config.output_dir, self.EXP_PATHS_DIR),
                )

    def run(self) -> None:
        """Executes the complete analysis pipeline."""
        self.load_data()
        self.create_output_dirs()
        self.run_fit_and_eval()
        self.analyze_expansion_paths()


def build_config_from_args() -> PerformanceFunctionConfig:
    """Builds the configuration object from command-line arguments."""
    from src.args import build_parser
    parser = build_parser()
    args = parser.parse_args()
    
    return PerformanceFunctionConfig(
        data_dir=args.data_dir,
        performance_file=args.performance_file,
        output_dir=args.output_dir,
        lang=args.lang,
        pivot_size=args.pivot_size,
        mode=args.mode.split(','),
        test_split_frac=args.test_split_frac,
        seed=args.seed,
        c12=args.c12
    )


def main():
    """Main entry point for the script."""
    config = build_config_from_args()
    analyzer = PerformanceFunctionAnalyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()
