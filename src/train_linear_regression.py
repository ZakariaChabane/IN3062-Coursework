# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:02:37 2025

@author: simpo
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing import SpotifyPopularityPreprocessor


SHOW_PLOTS = True  # set False for tests / no pop-up windows


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_split(name: str, y_true, y_pred) -> dict:
    return {
        f"{name}_rmse": rmse(y_true, y_pred),
        f"{name}_r2": float(r2_score(y_true, y_pred)),
    }


def main():
    # Preprocess (train/val/test)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # from src/ up to project root
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

    # Train Ridge Regression (regularized linear model) with CV
    alphas = np.logspace(-3, 3, 25)  # 0.001 -> 1000
    model = RidgeCV(alphas=alphas)
    model.fit(out.X_train, out.y_train)

    print("\nBest Ridge alpha:", float(model.alpha_))

    # Predict
    pred_train = model.predict(out.X_train)
    pred_val = model.predict(out.X_val)
    pred_test = model.predict(out.X_test)

    # Clip predictions to valid range (Spotify popularity is typically 0..100)
    pred_train = np.clip(pred_train, 0, 100)
    pred_val = np.clip(pred_val, 0, 100)
    pred_test = np.clip(pred_test, 0, 100)

    # Evaluate
    metrics = {}
    metrics.update(evaluate_split("train", out.y_train, pred_train))
    metrics.update(evaluate_split("val", out.y_val, pred_val))
    metrics.update(evaluate_split("test", out.y_test, pred_test))

    print("\n=== Ridge Regression Results ===")
    print(f"Train RMSE: {metrics['train_rmse']:.4f} | Train R²: {metrics['train_r2']:.4f}")
    print(f"Val   RMSE: {metrics['val_rmse']:.4f} | Val   R²: {metrics['val_r2']:.4f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.4f} | Test  R²: {metrics['test_r2']:.4f}")

    # Compare baseline mean predictor
    baseline = out.metadata.get("baseline_mean_predictor", {})
    if baseline:
        print("\n=== Mean Baseline (from preprocessing) ===")
        print(f"Mean value: {baseline.get('mean_value'):.4f}")
        print(f"Val  RMSE: {baseline.get('val_rmse'):.4f} | Val  R²: {baseline.get('val_r2'):.4f}")
        print(f"Test RMSE: {baseline.get('test_rmse'):.4f} | Test R²: {baseline.get('test_r2'):.4f}")

    # Save metrics + plots (use PROJECT_ROOT so paths are always correct)
    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(tables_dir / "linear_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot predicted vs Actual (Test)
    plt.figure()
    plt.scatter(out.y_test, pred_test, s=8)
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title("Ridge Regression: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig(figures_dir / "lr_actual_vs_pred_test.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Plot residuals (Test)
    residuals = out.y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals, s=8)
    plt.axhline(0)
    plt.xlabel("Predicted Popularity")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Ridge Regression: Residuals vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig(figures_dir / "lr_residuals_test.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Show top coefficients
    if out.feature_names and len(out.feature_names) == model.coef_.shape[0]:
        coef = model.coef_
        idx = np.argsort(np.abs(coef))[::-1][:15]
        print("\nTop 15 coefficients by absolute value:")
        for i in idx:
            print(f"{out.feature_names[i]:<25} {coef[i]: .5f}")

    # Save trained model
    try:
        import joblib

        artifact_dir = PROJECT_ROOT / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_dir / "linear_regression_model.joblib")
        print("\nSaved model to artifacts/linear_regression_model.joblib")
    except Exception as e:
        print("\nCould not save model (joblib missing or path issue):", e)


if __name__ == "__main__":
    main()
