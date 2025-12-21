# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 20:19:31 2025

@author: simpo
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing import SpotifyPopularityPreprocessor


def rmse(y_true, y_pred) -> float:
    """Version-proof RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    #Preprocess
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
    CSV_PATH = PROJECT_ROOT / "data" / "spotify_tracks.csv"
    
    print("Current working dir:", Path.cwd())
    print("Resolved CSV path:", CSV_PATH)
    print("CSV exists?:", CSV_PATH.exists())
    
    prep = SpotifyPopularityPreprocessor(
        csv_path=str(CSV_PATH),
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        scale_numeric=True,
        save_dir=str(PROJECT_ROOT / "artifacts"),
    )
    out = prep.run()

    #Hyperparameter search
    # Depth  model complexity
    # min_samples_leaf smooths prediction
    max_depth_grid = [3, 5, 8, 12, 16, 20, None]
    min_samples_leaf_grid = [1, 5, 10, 25, 50]

    best = None
    best_model = None
    results = []

    for max_depth in max_depth_grid:
        for min_leaf in min_samples_leaf_grid:
            model = DecisionTreeRegressor(
                random_state=42,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
            )
            model.fit(out.X_train, out.y_train)

            pred_train = model.predict(out.X_train)
            pred_val = model.predict(out.X_val)

            train_metrics = evaluate(out.y_train, pred_train)
            val_metrics = evaluate(out.y_val, pred_val)

            row = {
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
            }
            results.append(row)

            # Choose best using validation RMSE
            if best is None or row["val_rmse"] < best["val_rmse"]:
                best = row
                best_model = model

    print("\n=== Best Decision Tree (by validation RMSE) ===")
    print(best)

    #Final evaluation on test
    pred_train = best_model.predict(out.X_train)
    pred_val = best_model.predict(out.X_val)
    pred_test = best_model.predict(out.X_test)

    train_metrics = evaluate(out.y_train, pred_train)
    val_metrics = evaluate(out.y_val, pred_val)
    test_metrics = evaluate(out.y_test, pred_test)

    print("\n=== Decision Tree Results (best model) ===")
    print(f"Train RMSE: {train_metrics['rmse']:.4f} | Train R²: {train_metrics['r2']:.4f}")
    print(f"Val   RMSE: {val_metrics['rmse']:.4f} | Val   R²: {val_metrics['r2']:.4f}")
    print(f"Test  RMSE: {test_metrics['rmse']:.4f} | Test  R²: {test_metrics['r2']:.4f}")

    # Compare against baseline mean predictor
    baseline = out.metadata.get("baseline_mean_predictor", {})
    if baseline:
        print("\n=== Mean Baseline (from preprocessing) ===")
        print(f"Val  RMSE: {baseline.get('val_rmse'):.4f} | Val  R²: {baseline.get('val_r2'):.4f}")
        print(f"Test RMSE: {baseline.get('test_rmse'):.4f} | Test R²: {baseline.get('test_r2'):.4f}")

    #Save results + model
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Save full grid search results
    with open("results/tables/decision_tree_grid_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save best metrics
    best_summary = {
        "best_hyperparameters": {"max_depth": best["max_depth"], "min_samples_leaf": best["min_samples_leaf"]},
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open("results/tables/decision_tree_best_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    # Save the model
    try:
        import joblib
        joblib.dump(best_model, "artifacts/decision_tree_model.joblib")
        print("\nSaved model to artifacts/decision_tree_model.joblib")
    except Exception as e:
        print("\nCould not save model:", e)

    #Plots for report

    # Predicted vs Actual (Test)
    plt.figure()
    plt.scatter(out.y_test, pred_test, s=8)
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title("Decision Tree: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("results/figures/dt_actual_vs_pred_test.png", dpi=300)
    plt.show()

    # Residuals vs Predicted (Test)
    residuals = out.y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals, s=8)
    plt.axhline(0)
    plt.xlabel("Predicted Popularity")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Decision Tree: Residuals vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("results/figures/dt_residuals_test.png", dpi=300)
    plt.show()

    # Depth vs Validation RMSE plot
    # Aggregate best val_rmse per depth
    depth_to_best_val = {}
    for row in results:
        d = row["max_depth"]
        depth_to_best_val[d] = min(depth_to_best_val.get(d, float("inf")), row["val_rmse"])

    # Convert to sorted lists
    depths = [d for d in depth_to_best_val.keys() if d is not None]
    depths = sorted(depths)
    if None in depth_to_best_val:
        depths.append(None)

    xs = [(-1 if d is None else d) for d in depths]  # plot None as -1
    ys = [depth_to_best_val[d] for d in depths]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("max_depth (None shown as -1)")
    plt.ylabel("Best Validation RMSE")
    plt.title("Decision Tree: Model Complexity vs Validation Error")
    plt.tight_layout()
    plt.savefig("results/figures/dt_depth_vs_val_rmse.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
