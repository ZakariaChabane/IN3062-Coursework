# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:40:38 2025

@author: simpo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib


@dataclass
class PreprocessOutput:
    """Outputs required to train/evaluate models consistently."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    metadata: dict


class SpotifyPopularityPreprocessor:

    # Columns not used
    DROP_COLS = ["Unnamed: 0", "track_id", "track_name", "album_name", "artists", "track_genre"]

    # Categorical columns to encode
    CAT_COLS = ["explicit", "key", "mode", "time_signature"]

    def __init__(
        self,
        csv_path: str,
        target_col: str = "popularity",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        scale_numeric: bool = True,
        save_dir: Optional[str] = "artifacts",
    ) -> None:
        self.csv_path = csv_path
        self.target_col = target_col
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.scale_numeric = bool(scale_numeric)
        self.save_dir = save_dir
        self._pipeline: Optional[ColumnTransformer] = None
        self._feature_names: Optional[List[str]] = None
        self._numeric_cols: Optional[List[str]] = None

    def run(self) -> PreprocessOutput:
        """
        Full preprocessing run:
          1) load data
          2) basic cleaning
          3) define X/y
          4) train/val/test split
          5) fit preprocessing on train only
          6) transform splits
          7) compute simple baselines
        """
        df = self._load()
        df = self._basic_clean(df)

        X, y = self._make_X_y(df)

        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = self._split(X, y)

        self._numeric_cols = self._infer_numeric_cols(X_train_raw)

        self._pipeline = self._build_pipeline(self._numeric_cols, self.CAT_COLS)
        self._pipeline.fit(X_train_raw)  # FIT ONLY ON TRAIN

        X_train = self._to_dense(self._pipeline.transform(X_train_raw))
        X_val = self._to_dense(self._pipeline.transform(X_val_raw))
        X_test = self._to_dense(self._pipeline.transform(X_test_raw))

        self._feature_names = self._get_feature_names()

        # sanity checks
        baseline = self._compute_mean_baseline(y_train, y_val, y_test)

        metadata = {
            "csv_path": self.csv_path,
            "target_col": self.target_col,
            "dropped_columns": self.DROP_COLS,
            "categorical_columns": self.CAT_COLS,
            "numeric_columns_used": self._numeric_cols,
            "split": {
                "train_frac": 1.0 - (self.test_size + self.val_size),
                "val_frac": self.val_size,
                "test_frac": self.test_size,
                "random_state": self.random_state,
            },
            "baseline_mean_predictor": baseline,
        }

        if self.save_dir is not None:
            self._save_artifacts(metadata)

        return PreprocessOutput(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=self._feature_names,
            metadata=metadata,
        )

    def transform_new(self, df_new: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Run .run() first.")
        df_new = self._basic_clean(df_new)
        if self.target_col in df_new.columns:
            df_new = df_new.drop(columns=[self.target_col], errors="ignore")
        X_new = df_new.drop(columns=[c for c in self.DROP_COLS if c in df_new.columns], errors="ignore")
        return self._to_dense(self._pipeline.transform(X_new))


    def _load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in CSV columns.")
        return df

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal cleaning:
            Drop duplicate rows
            Drop rows with missing target
        """
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)
        return df

    def _make_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        y = df[self.target_col].to_numpy(dtype=float)
        X = df.drop(columns=[self.target_col], errors="ignore")

        # Drop identifier columns
        X = X.drop(columns=[c for c in self.DROP_COLS if c in X.columns], errors="ignore")

        return X, y

    def _split(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split:
            train vs temp
            temp -> val/test
        """
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.test_size + self.val_size),
            random_state=self.random_state,
            shuffle=True
        )

        test_prop = self.test_size / (self.test_size + self.val_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_prop,
            random_state=self.random_state,
            shuffle=True
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _infer_numeric_cols(self, X_train: pd.DataFrame) -> List[str]:
        """
        Numeric columns = all numeric columns except catagorical
        """
        numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
        numeric = [c for c in numeric if c not in self.CAT_COLS]
        return numeric

    def _build_pipeline(self, numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        """
        Preprocessing pipeline:
            Numeric - median impute +  standardize
            Categorical - most-frequent impute + one-hot
        """
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(steps=num_steps)

        # One-hot encoding
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe)
        ])

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _get_feature_names(self) -> List[str]:
        if self._pipeline is None:
            return []
        try:
            return self._pipeline.get_feature_names_out().tolist()
        except Exception:
            return []

    @staticmethod
    def _to_dense(arr) -> np.ndarray:
        # Handle numpy array already input
        return np.asarray(arr)

    @staticmethod
    def _compute_mean_baseline(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Mean predictor baseline:
        predict mean(y_train); compute RMSE and R^2 for val/test.
        """
        mean_pred = float(np.mean(y_train))

        val_pred = np.full(shape=y_val.shape, fill_value=mean_pred, dtype=float)
        test_pred = np.full(shape=y_test.shape, fill_value=mean_pred, dtype=float)

        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        return {
            "mean_value": mean_pred,
            "val_rmse": float(np.sqrt(val_mse)),
            "val_r2": float(r2_score(y_val, val_pred)),
            "test_rmse": float(np.sqrt(test_mse)),
            "test_r2": float(r2_score(y_test, test_pred)),
        }

    def _save_artifacts(self, metadata: dict) -> None:
        """
        Saves:
            preprocessing pipeline as joblib
            metadata.json describing preprocessing choices
        """
        os.makedirs(self.save_dir, exist_ok=True)

        if self._pipeline is not None:
            joblib.dump(self._pipeline, os.path.join(self.save_dir, "preprocess_pipeline.joblib"))

        with open(os.path.join(self.save_dir, "preprocess_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
