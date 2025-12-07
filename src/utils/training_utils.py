import pandas as pd

def analyze_cv_results(cv_results: pd.DataFrame, metric_name: str = 'test-quantile-mean'):
    """
    Analyzes cross-validation results to find the optimal number of rounds and the best score.

    Args:
        cv_results (pd.DataFrame): The results DataFrame from xgb.cv.
        metric_name (str): The name of the metric to analyze.

    Returns:
        tuple: (best_round, best_score)
    """
    if metric_name not in cv_results.columns:
        raise ValueError(f"Metric {metric_name} not found in CV results columns: {cv_results.columns}")
        
    best_round = cv_results[metric_name].argmin() + 1
    best_score = cv_results[metric_name].min()
    
    return best_round, best_score

def format_training_start(model_name: str, has_console: bool = False) -> str:
    """Returns the formatted string for training start."""
    if has_console:
        return f"Training [bold]{model_name}[/bold]..."
    return f"Training {model_name}..."

def format_cv_start(model_name: str, has_console: bool = False) -> str:
    """Returns the formatted string for CV start."""
    return f"Running Cross-Validation for {model_name}..."

def format_cv_results(best_round: int, metric_name: str, best_score: float, has_console: bool = False) -> str:
    """Returns the formatted string for CV results."""
    if has_console:
        return f"[cyan]Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}[/cyan]"
    return f"Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}"

def format_final_training(best_round: int, has_console: bool = False) -> str:
    """Returns the formatted string for final training start."""
    if has_console:
        return f"[dim]Training final model with {best_round} rounds...[/dim]"
    return f"Training final model with {best_round} rounds..."
