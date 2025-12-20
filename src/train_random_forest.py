# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:24:11 2025

@author: simpo
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing import SpotifyPopularityPreprocessor


def rmse(y_true, y_pred) -> float:
    """Version-safe RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():

    # Preprocess data    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CSV_PATH = PROJECT_ROOT / "data" / "spotify_tracks.csv"
    
    print("Current working dir:", Path.cwd())
    print("Resolved CSV path:", CSV_PATH)
    print("CSV exists?:", CSV_PATH.exists())
    
    prep = SpotifyPopularityPreprocessor(
        csv_path=str(CSV_PATH),
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        scale_numeric=True,
        save_dir=str(PROJECT_ROOT / "artifacts"),
    )
    out = prep.run()

    #Hyperparameter grid
    n_estimators_grid = [10, 50, 100, 200, 300, 400]
    max_depth_grid = [2, 5 ,7 ,10, 12, 15, 17, 20, 25, None]
    min_samples_leaf_grid = [1, 2, 3, 5, 10, 25]
    max_features_grid = [0.2, 0.5, 1.0, None, "sqrt", "log2"]
    min_weight_fraction_leaf_grid = [0.0, 0.2, 0.5]

    best = None
    best_model = None
    results = []

    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            for min_leaf in min_samples_leaf_grid:
                for max_features in max_features_grid:
                    for min_weight_fraction_leaf in min_weight_fraction_leaf_grid:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_leaf,
                            max_features = max_features,
                            min_weight_fraction_leaf = min_weight_fraction_leaf,
                            random_state=42,
                            n_jobs=-1,
                        )

                        model.fit(out.X_train, out.y_train)
        
                        pred_train = model.predict(out.X_train)
                        pred_val = model.predict(out.X_val)
        
                        train_metrics = evaluate(out.y_train, pred_train)
                        val_metrics = evaluate(out.y_val, pred_val)
        
                        row = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_features,
                            "min_weight_fraction_leaf" : min_weight_fraction_leaf,
                            "train_rmse": train_metrics["rmse"],
                            "train_r2": train_metrics["r2"],
                            "val_rmse": val_metrics["rmse"],
                            "val_r2": val_metrics["r2"],
                        }
                        results.append(row)
        
                        if best is None or row["val_rmse"] < best["val_rmse"]:
                            best = row
                            best_model = model

    print("\n=== Best Random Forest (by validation RMSE) ===")
    print(best)

    #Final evaluation on test
    pred_train = best_model.predict(out.X_train)
    pred_val = best_model.predict(out.X_val)
    pred_test = best_model.predict(out.X_test)

    train_metrics = evaluate(out.y_train, pred_train)
    val_metrics = evaluate(out.y_val, pred_val)
    test_metrics = evaluate(out.y_test, pred_test)

    print("\n=== Random Forest Results (best model) ===")
    print(f"Train RMSE: {train_metrics['rmse']:.4f} | Train R²: {train_metrics['r2']:.4f}")
    print(f"Val   RMSE: {val_metrics['rmse']:.4f} | Val   R²: {val_metrics['r2']:.4f}")
    print(f"Test  RMSE: {test_metrics['rmse']:.4f} | Test  R²: {test_metrics['r2']:.4f}")


    #Save results + model
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    with open("results/tables/random_forest_grid_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = {
        "best_hyperparameters": {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "min_samples_leaf": best["min_samples_leaf"],
            "max_features": best["max_features"],
            "min_weight_fraction_leaf": best["min_weight_fraction_leaf"],
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open("results/tables/random_forest_best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import joblib
        joblib.dump(best_model, "artifacts/random_forest_model.joblib")
        print("\nSaved model to artifacts/random_forest_model.joblib")
    except Exception as e:
        print("\nCould not save model:", e)

    # ----------------------------
    # Plots
    # Actual vs Predicted (Test)
    plt.figure()
    plt.scatter(out.y_test, pred_test, s=8)
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title("Random Forest: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("results/figures/rf_actual_vs_pred_test.png", dpi=300)
    plt.show()

    # Residuals (Test)
    residuals = out.y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals, s=8)
    plt.axhline(0)
    plt.xlabel("Predicted Popularity")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Random Forest: Residuals vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("results/figures/rf_residuals_test.png", dpi=300)
    plt.show()

    # Feature importance plot
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        idx = np.argsort(importances)[::-1][:15]

        plt.figure()
        plt.barh(
            range(len(idx)),
            importances[idx][::-1],
            align="center"
        )
        labels = (
            [out.feature_names[i] for i in idx][::-1]
            if out.feature_names
            else [f"f{i}" for i in idx][::-1]
        )
        plt.yticks(range(len(idx)), labels)
        plt.xlabel("Feature Importance")
        plt.title("Random Forest: Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig("results/figures/rf_feature_importance.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
