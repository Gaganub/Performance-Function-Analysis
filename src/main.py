import os
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Dict, Tuple, List, Optional, Union, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s"
)
logger = logging.getLogger("PFA")
logger.setLevel(logging.INFO)


@dataclass
class PerformanceFunctionConfig:
    data_dir: str
    performance_file: str
    output_dir: str
    lang: str
    pivot_size: str
    mode: List[str]
    test_split_frac: float
    seed: int
    c12: float


class PerformanceFunctionAnalyzer:
    def __init__(self, config: PerformanceFunctionConfig):
        self.config = config
        self.perf_df = None
        self.langs = []
        self.pivot_sizes = []
        
    def load_data(self) -> None:
        """Load performance data from file"""
        perf_filepath = os.path.join(self.config.data_dir, self.config.performance_file)
        logger.info(f"Loading Data from {perf_filepath}")
        self.perf_df = pd.read_csv(perf_filepath)
        
        # Set languages and pivot sizes
        self.langs = ([self.config.lang] if self.config.lang != "all" 
                      else sorted(list(self.perf_df["tgt_lang"].unique())))
        self.pivot_sizes = ([int(self.config.pivot_size)] if self.config.pivot_size != "all" 
                           else sorted(list(self.perf_df["en_pivot_size"].unique())))
        
        # Filter data
        self.perf_df = self.perf_df[
            (self.perf_df["tgt_lang"].isin(self.langs)) & 
            (self.perf_df["en_pivot_size"].isin(self.pivot_sizes))
        ]
        
    def create_output_dirs(self) -> None:
        """Create necessary output directories"""
        logger.info("Creating Output Directories if not exists")
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
            
        if "fit_nd_eval" in self.config.mode:
            pred_dir = os.path.join(self.config.output_dir, "fit_results")
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
                
        if "expansion_paths" in self.config.mode:
            exp_path_dir = os.path.join(self.config.output_dir, "exp_paths")
            if not os.path.exists(exp_path_dir):
                os.makedirs(exp_path_dir)
                
    def fit_and_evaluate(self, model_type: str, train_df: pd.DataFrame, 
                         test_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict, float, float, float]:
        """Fit and evaluate a performance function model
        
        Args:
            model_type: Type of model - "amue" or "gpr"
            train_df: Training data
            test_df: Test data (optional)
            
        Returns:
            Tuple containing:
                - DataFrame with predictions and errors
                - Dictionary of model parameters
                - Mean Absolute Error (MAE)
                - Mean Squared Error (MSE)
                - R² score
        """
        pred_error_dfs = []
        lang2ps2parms = {}

        for lang, lang_sub_train_df in train_df.groupby("tgt_lang"):
            if lang == "en":
                continue
                
            lang2ps2parms[lang] = {}
            for pivot_size, lang_ps_sub_train_df in lang_sub_train_df.groupby("en_pivot_size"):
                print(f"Lang: {lang}, Pivot Size: {pivot_size}, Length of Training Data: {len(lang_ps_sub_train_df)}")
                
                try:
                    if test_df is not None:
                        lang_ps_sub_test_df = test_df[
                            (test_df["tgt_lang"] == lang) & 
                            (test_df["en_pivot_size"] == pivot_size)
                        ]
                        
                        if model_type == "amue":
                            from src.fit_performance_functions import fit_nd_eval_amue_model_diff_test
                            params, pred_nd_error_df = fit_nd_eval_amue_model_diff_test(
                                lang_ps_sub_train_df, lang_ps_sub_test_df, lang, pivot_size
                            )
                        else:
                            from src.fit_performance_functions import fit_nd_eval_gpr_model_diff_test
                            params, pred_nd_error_df = fit_nd_eval_gpr_model_diff_test(
                                lang_ps_sub_train_df, lang_ps_sub_test_df, lang, pivot_size
                            )
                    else:
                        if model_type == "amue":
                            from src.fit_performance_functions import fit_nd_eval_amue_model
                            params, pred_nd_error_df = fit_nd_eval_amue_model(
                                lang_ps_sub_train_df, lang, pivot_size
                            )
                        else:
                            from src.fit_performance_functions import fit_nd_eval_gpr_model
                            params, pred_nd_error_df = fit_nd_eval_gpr_model(
                                lang_ps_sub_train_df, lang, pivot_size
                            )
                            
                except ValueError:
                    continue
                    
                pred_error_dfs.append(pred_nd_error_df)
                
                if isinstance(params, np.ndarray):
                    lang2ps2parms[lang][pivot_size] = params.tolist()
                else:
                    lang2ps2parms[lang][pivot_size] = params

        pred_error_df = pd.concat(pred_error_dfs, axis=0)
        mae = pred_error_df["Absolute Errors"].mean()
        mse = pred_error_df["Squared Errors"].mean() ** (1 / 2)
        r2 = r2_score(
            pred_error_df["F1-Score"].values, pred_error_df["Predicted F1-Score"].values,
        )

        return pred_error_df, lang2ps2parms, mae, mse, r2
    
    def run_fit_and_eval(self) -> None:
        """Run fitting and evaluation process"""
        if "fit_nd_eval" not in self.config.mode:
            return
            
        logger.info("Splitting Performance data into train and test data")
        perf_train_df, perf_test_df = train_test_split(
            self.perf_df, test_size=self.config.test_split_frac, random_state=self.config.seed
        )

        # Fit and evaluate AMUE model
        logger.info("Fitting And Evaluating AMUE Performance Function")
        amue_results = self.fit_and_evaluate("amue", perf_train_df, perf_test_df)
        amue_pred_error_df, amue_lang2ps2parms, amue_mae, amue_mse, amue_r2 = amue_results
        
        logger.info(f"Done Fitting and Evaluating AMUE | MAE: {amue_mae} | RMSE: {amue_mse} | R^2: {amue_r2}")

        # Save predictions and errors
        logger.info("Saving Prediction and Errors")
        pred_dir = os.path.join(self.config.output_dir, "fit_results")
        pred_file = os.path.join(
            pred_dir,
            f"amue_pred_nd_errors_lang{self.config.lang}_pivotSize{self.config.pivot_size}.csv",
        )
        amue_pred_error_df.to_csv(pred_file)

        # Save parameters
        logger.info("Saving Parameters")
        params_file = os.path.join(
            pred_dir, f"amue_params_lang{self.config.lang}_pivotSize{self.config.pivot_size}.json"
        )
        with open(params_file, "w") as f:
            json.dump(amue_lang2ps2parms, f, indent=4, ensure_ascii=False)

        # Fit and evaluate GPR model
        logger.info("Fitting And Evaluating GPR Performance Function")
        _, _, gpr_mae, gpr_mse, gpr_r2 = self.fit_and_evaluate("gpr", perf_train_df, perf_test_df)
        logger.info(f"Done Fitting and Evaluating GPR | MAE: {gpr_mae} | MSE: {gpr_mse} | R^2: {gpr_r2}")
        
    def analyze_expansion_paths(self) -> None:
        """Analyze expansion paths"""
        if "expansion_paths" not in self.config.mode:
            return
            
        logger.info("Fitting AMUE Performance Function on entire dataset")
        _, amue_lang2ps2parms, _, _, _ = self.fit_and_evaluate("amue", self.perf_df)
        
        from src.perf_func_analysis import get_expansion_path
        
        for lang in self.langs:
            if lang == "en":
                continue
                
            for pivot_size in self.pivot_sizes:
                if pivot_size == 0:
                    continue
                    
                params = amue_lang2ps2parms[lang][pivot_size]
                a0 = params[0]
                max_y = int(
                    self.perf_df[
                        (self.perf_df["tgt_lang"] == lang) & 
                        (self.perf_df["en_pivot_size"] == pivot_size)
                    ]["f1_score"].max()
                )
                ys = np.linspace(np.ceil(a0), max_y, int((max_y - a0) // 1))
                
                get_expansion_path(
                    params,
                    ys,
                    self.config.c12,
                    lang,
                    pivot_size,
                    plot=True,
                    save_dir=os.path.join(self.config.output_dir, "exp_paths"),
                )
    
    def run(self) -> None:
        """Run the performance function analysis"""
        self.load_data()
        self.create_output_dirs()
        self.run_fit_and_eval()
        self.analyze_expansion_paths()


def build_config_from_args() -> PerformanceFunctionConfig:
    """Build configuration from command line arguments"""
    from src.args import build_parser
    args = build_parser()
    
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
    """Main entry point"""
    config = build_config_from_args()
    analyzer = PerformanceFunctionAnalyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()
