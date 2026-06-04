from pathlib import Path
import pandas as pd
from config import EXPERIMENTS, RANDOM_SEED
from data_generator import make_two_moons, train_test_split
from training import train_model

def run_all(results_dir: Path):
    X, y = make_two_moons()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    histories = []
    models = {}
    for i, exp in enumerate(EXPERIMENTS):
        model, hist = train_model(X_train, y_train, X_test, y_test, exp, seed=RANDOM_SEED + i)
        histories.append(hist)
        models[exp["name"]] = model
    history_df = pd.concat(histories, ignore_index=True)
    final_df = history_df.sort_values("epoch").groupby("experiment").tail(1).copy()
    final_df = final_df[["experiment", "learning_rate", "hidden_size", "activation", "train_loss", "test_loss", "train_accuracy", "test_accuracy", "gradient_norm", "update_norm"]]
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, columns=["x1", "x2"]).assign(label=y.ravel()).to_csv(results_dir / "dataset.csv", index=False)
    history_df.to_csv(results_dir / "training_history.csv", index=False)
    final_df.to_csv(results_dir / "summary_results.csv", index=False)
    return X_train, X_test, y_train, y_test, models, history_df, final_df
