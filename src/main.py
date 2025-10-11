import os
import logging
import json
import argparse
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PerformanceFunctionConfig":
        """Factory method to create a config instance from parsed arguments."""
        return cls(
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


class ModelEvaluationResults(NamedTuple):
    """Data structure for holding model evaluation results."""
    pred_error_df: pd.DataFrame
    params: Dict[str, Any]
    mae: float
    rmse: float
    r2: float


class PerformanceFunctionAnalyzer:
    """A class to analyze performance functions by fitting and evaluating models."""
    TGT_LANG_COL = "tgt_lang"
    PIVOT_SIZE_COL = "en_pivot_size"
    F1_SCORE_COL = "F1-Score"
    PREDICTED_F1_COL = "Predicted F1-Score"
    ABS_ERROR_COL = "Absolute Errors"
    SQ_ERROR_COL = "Squared Errors"
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
        try:
            self.perf_df = pd.read_csv(perf_filepath)
        except FileNotFoundError:
            logger.error(f"Performance file not found at {perf_filepath}. Exiting.")
            raise

        self.langs = sorted(self.perf_df[self.TGT_LANG_COL].unique()) if self.config.lang == "all" else [self.config.lang]
        self.pivot_sizes = sorted(self.perf_df[self.PIVOT_SIZE_COL].unique()) if self.config.pivot_size == "all" else [int(self.config.pivot_size)]

        mask = (self.perf_df[self.TGT_LANG_COL].isin(self.langs)) & (self.perf_df[self.PIVOT_SIZE_COL].isin(self.pivot_sizes))
        self.perf_df = self.perf_df[mask].copy()
        logger.info(f"Data loaded and filtered. Final shape: {self.perf_df.shape}")

    def create_output_dirs(self) -> None:
        """Creates the necessary output directories for saving results."""
        logger.info("Creating output directories...")
        os.makedirs(self.config.output_dir, exist_ok=True)
        if "fit_nd_eval" in self.config.mode:
            os.makedirs(os.path.join(self.config.output_dir, self.FIT_RESULTS_DIR), exist_ok=True)
        if "expansion_paths" in self.config.mode:
            os.makedirs(os.path.join(self.config.output_dir, self.EXP_PATHS_DIR), exist_ok=True)

    def _fit_model(self, model_type: str, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame], lang: str, pivot_size: int) -> Tuple[Any, pd.DataFrame]:
        """Helper method to dynamically import and call the correct model fitting function."""
        from src.fit_performance_functions import (
            fit_nd_eval_amue_model_diff_test, fit_nd_eval_gpr_model_diff_test,
            fit_nd_eval_amue_model, fit_nd_eval_gpr_model
        )
        
        if test_df is not None:
            model_func = fit_nd_eval_amue_model_diff_test if model_type == "amue" else fit_nd_eval_gpr_model_diff_test
            return model_func(train_df, test_df, lang, pivot_size)
        else:
            model_func = fit_nd_eval_amue_model if model_type == "amue" else fit_nd_eval_gpr_model
            return model_func(train_df, lang, pivot_size)

    def fit_and_evaluate(self, model_type: str, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> ModelEvaluationResults:
        """Fits and evaluates a performance function model."""
        pred_error_dfs = []
        lang2ps2parms = {}

        for lang, lang_sub_train_df in train_df.groupby(self.TGT_LANG_COL):
            if lang == "en": continue
            
            lang2ps2parms[lang] = {}
            for pivot_size, lang_ps_sub_train_df in lang_sub_train_df.groupby(self.PIVOT_SIZE_COL):
                logger.info(f"Processing {model_type.upper()} for Lang: {lang}, Pivot: {pivot_size}, Train Size: {len(lang_ps_sub_train_df)}")
                
                try:
                    sub_test_df = test_df[(test_df[self.TGT_LANG_COL] == lang) & (test_df[self.PIVOT_SIZE_COL] == pivot_size)] if test_df is not None else None
                    params, pred_nd_error_df = self._fit_model(model_type, lang_ps_sub_train_df, sub_test_df, lang, pivot_size)
                    pred_error_dfs.append(pred_nd_error_df)
                    lang2ps2parms[lang][pivot_size] = params.tolist() if isinstance(params, np.ndarray) else params
                except (ValueError, ImportError, KeyError) as e:
                    logger.warning(f"Could not fit model for {lang}-{pivot_size}. Error: {e}")
                    continue

        if not pred_error_dfs:
            return ModelEvaluationResults(pd.DataFrame(), {}, np.nan, np.nan, np.nan)

        pred_error_df = pd.concat(pred_error_dfs, ignore_index=True)
        mae = pred_error_df[self.ABS_ERROR_COL].mean()
        rmse = np.sqrt(pred_error_df[self.SQ_ERROR_COL].mean())
        r2 = r2_score(pred_error_df[self.F1_SCORE_COL], pred_error_df[self.PREDICTED_F1_COL])

        return ModelEvaluationResults(pred_error_df, lang2ps2parms, mae, rmse, r2)

    def run_fit_and_eval(self) -> None:
        """Runs the full fitting and evaluation process for configured models."""
        if "fit_nd_eval" not in self.config.mode: return
            
        logger.info(f"Splitting data (test size: {self.config.test_split_frac}, seed: {self.config.seed})...")
        perf_train_df, perf_test_df = train_test_split(self.perf_df, test_size=self.config.test_split_frac, random_state=self.config.seed)

        for model_name in ["amue", "gpr"]:
            logger.info(f"--- Starting {model_name.upper()} Model Evaluation ---")
            results = self.fit_and_evaluate(model_name, perf_train_df, perf_test_df)
            logger.info(f"{model_name.upper()} Evaluation Complete | MAE: {results.mae:.4f} | RMSE: {results.rmse:.4f} | R²: {results.r2:.4f}")
             
            if model_name == "amue" and not results.pred_error_df.empty:
                logger.info("Saving AMUE model results...")
                pred_dir = os.path.join(self.config.output_dir, self.FIT_RESULTS_DIR)
                base_filename = f"amue_lang_{self.config.lang}_pivot_{self.config.pivot_size}"
                results.pred_error_df.to_csv(os.path.join(pred_dir, f"{base_filename}_preds.csv"), index=False)
                with open(os.path.join(pred_dir, f"{base_filename}_params.json"), "w") as f:
                    json.dump(results.params, f, indent=4, ensure_ascii=False)

    def analyze_expansion_paths(self) -> None:
        """Analyzes and plots the expansion paths based on the fitted AMUE model."""
        if "expansion_paths" not in self.config.mode: return
            
        logger.info("Fitting AMUE on all data for expansion path analysis...")
        results = self.fit_and_evaluate("amue", self.perf_df)
        if not results.params:
            logger.error("Failed to fit AMUE model for expansion paths. Aborting.")
            return

        from src.perf_func_analysis import get_expansion_path
        max_f1_scores = self.perf_df.groupby([self.TGT_LANG_COL, self.PIVOT_SIZE_COL])[self.F1_SCORE_COL].max()
        
        for (lang, pivot_size), max_y in max_f1_scores.items():
            if lang == "en" or pivot_size == 0: continue
            
            params = results.params.get(lang, {}).get(pivot_size)
            if not params: continue

            a0 = params[0]
            if max_y <= np.ceil(a0): continue
            
            ys = np.linspace(np.ceil(a0), max_y, int(max_y - np.ceil(a0)) + 1)
            get_expansion_path(params, ys, self.config.c12, lang, pivot_size, plot=True, save_dir=os.path.join(self.config.output_dir, self.EXP_PATHS_DIR))

    def run(self) -> None:
        """Executes the complete analysis pipeline."""
        self.load_data()
        self.create_output_dirs()
        self.run_fit_and_eval()
        self.analyze_expansion_paths()

def main():
    """Main entry point for the script."""
    # Assuming build_parser is in src.args as per the original code
    from src.args import build_parser 
    parser = build_parser()
    args = parser.parse_args()
    
    config = PerformanceFunctionConfig.from_args(args)
    analyzer = PerformanceFunctionAnalyzer(config)
    analyzer.run()

if __name__ == "__main__":
    main()
