# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:53:15 2025

@author: zakaria
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.preprocessing import SpotifyPopularityPreprocessor


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_mlp(input_dim: int, hidden_layers: list[int], dropout: float, lr: float) -> tf.keras.Model:
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))

    if dropout > 0:
        model.add(Dropout(dropout))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
        if dropout > 0:
            model.add(Dropout(dropout))

    # Regression output
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[],
    )
    return model


def main():
    #Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    #Preprocess
    PROJECT_ROOT = Path(__file__).resolve().parents[0]  # goes from src/ up to project root
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

    X_train, y_train = out.X_train, out.y_train
    X_val, y_val = out.X_val, out.y_val
    X_test, y_test = out.X_test, out.y_test

    input_dim = X_train.shape[1]

    #Architechture search
    configs = [
        {"name": "Opt_Wide", "hidden_layers": [512, 256, 128], "dropout": 0.3, "lr": 0.0005, "batch_size": 128},
        {"name": "Opt_Deep", "hidden_layers": [256, 256, 128, 64], "dropout": 0.2, "lr": 0.0003, "batch_size": 64},
        {"name": "Opt_Fast", "hidden_layers": [256, 128], "dropout": 0.1, "lr": 0.001, "batch_size": 32},
    ]


    best_cfg = None
    best_model = None
    best_history = None
    best_val_rmse = None
    all_results = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\n=== Training config {i}/{len(configs)}: {cfg} ===")

        model = build_mlp(
            input_dim=input_dim,
            hidden_layers=cfg["hidden_layers"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=25,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=400,
            batch_size=cfg["batch_size"],
            verbose=1,
            callbacks=callbacks,
        )

        # Evaluate on val
        pred_val = model.predict(X_val, verbose=0).reshape(-1)
        val_metrics = evaluate(y_val, pred_val)

        row = {
            "name": cfg["name"],
            "hidden_layers": cfg["hidden_layers"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "best_val_loss": float(np.min(history.history["val_loss"])),
            "epochs_ran": len(history.history["loss"]),
        }

        all_results.append(row)

        print(f"Validation RMSE: {row['val_rmse']:.4f} | Validation R²: {row['val_r2']:.4f}")

        if best_val_rmse is None or row["val_rmse"] < best_val_rmse:
            best_val_rmse = row["val_rmse"]
            best_cfg = cfg
            best_model = model
            best_history = history

    print("\n=== Best MLP config (by validation RMSE) ===")
    print(best_cfg)
    print(f"Best validation RMSE: {best_val_rmse:.4f}")

    #Final evaluation on train/val/test
    pred_train = best_model.predict(X_train, verbose=0).reshape(-1)
    pred_val = best_model.predict(X_val, verbose=0).reshape(-1)
    pred_test = best_model.predict(X_test, verbose=0).reshape(-1)

    train_metrics = evaluate(y_train, pred_train)
    val_metrics = evaluate(y_val, pred_val)
    test_metrics = evaluate(y_test, pred_test)

    print("\n=== MLP Results (best model) ===")
    print(f"Train RMSE: {train_metrics['rmse']:.4f} | Train R²: {train_metrics['r2']:.4f}")
    print(f"Val   RMSE: {val_metrics['rmse']:.4f} | Val   R²: {val_metrics['r2']:.4f}")
    print(f"Test  RMSE: {test_metrics['rmse']:.4f} | Test  R²: {test_metrics['r2']:.4f}")

    #Save outputs
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    summary = {
        "best_config": best_cfg,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "all_results": all_results,
    }

    with open("results/tables/mlp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save model
    best_model.save("artifacts/mlp_model.keras")
    print("\nSaved model to artifacts/mlp_model.keras")

    #Plots for report
    # Learning curves (loss)
    plt.figure()
    plt.plot(best_history.history["loss"], label="train_loss")
    plt.plot(best_history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training Curve (Best Config)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/mlp_learning_curve.png", dpi=300)
    plt.show()

    # Actual vs Predicted (Test)
    plt.figure()
    plt.scatter(y_test, pred_test, s=8)
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title("MLP: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("results/figures/mlp_actual_vs_pred_test.png", dpi=300)
    plt.show()

    # Residuals plot (Test)
    residuals = y_test - pred_test
    plt.figure()
    plt.scatter(pred_test, residuals, s=8)
    plt.axhline(0)
    plt.xlabel("Predicted Popularity")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("MLP: Residuals vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("results/figures/mlp_residuals_test.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
