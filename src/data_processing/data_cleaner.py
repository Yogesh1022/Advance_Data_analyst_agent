import polars as pl
import numpy as np
from scipy import stats
import re

class DataCleaner:
    def run_pipeline(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Executes a data cleaning pipeline on the Polars DataFrame.
        Steps include: basic quality checks, data type standardization,
        handling missing values, and outlier detection/treatment.
        """
        self.initial_checks(df)
        df = self.standardize_data_types(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        self.final_validation(df)
        return df

    def initial_checks(self, df: pl.DataFrame):
        """Performs basic data quality checks."""
        print(f"Initial Data Info:\n{df.schema}")
        print(f"\nMissing Values:\n{df.select(pl.col('*').is_null().sum())}")
        print(f"\nDuplicate Rows: {df.is_duplicated().sum()}")
        print(f"\nDescriptive Statistics:\n{df.describe()}")

    def standardize_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardizes data types and cleans column names."""
        # Clean column names (lowercase, replace spaces with underscores)
        cleaned_columns = [
            re.sub(r'\W+', '_', col.lower().strip()) for col in df.columns
        ]
        df = df.rename(dict(zip(df.columns, cleaned_columns)))

        # Convert string columns to numeric if possible
        for col in df.select(pl.col(pl.Utf8)).columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            except pl.exceptions.ComputeError:
                pass  # Skip if casting fails

        # Convert date-like columns to datetime
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
                except pl.exceptions.ComputeError:
                    pass
        return df

    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handles missing values using imputation strategies."""
        for col in df.columns:
            if df.select(pl.col(col).is_null().sum()).item() > 0:
                if df[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                    # Impute numerical columns with median
                    median_val = df.select(pl.col(col).median()).item()
                    df = df.with_columns(pl.col(col).fill_null(median_val))
                    print(f"Filled missing values in '{col}' with median: {median_val}")
                elif df[col].dtype == pl.Utf8 or df[col].dtype.is_in([pl.Categorical, pl.Enum]):
                    # Impute categorical/string columns with mode
                    mode_val = df.select(pl.col(col).mode()).item()
                    if mode_val is not None:
                        df = df.with_columns(pl.col(col).fill_null(mode_val))
                        print(f"Filled missing values in '{col}' with mode: {mode_val}")
                    else:
                        print(f"No mode found for '{col}', skipping imputation.")
                else:
                    # For other types, drop rows
                    df = df.filter(~pl.col(col).is_null())
                    print(f"Dropped rows with missing values in '{col}'")
        return df

    def handle_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detects and handles outliers using the IQR method (capping)."""
        for col in df.select(pl.col(pl.Float64, pl.Int64)).columns:
            q1 = df.select(pl.col(col).quantile(0.25)).item()
            q3 = df.select(pl.col(col).quantile(0.75)).item()
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_count = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).height
            if outliers_count > 0:
                print(f"Found {outliers_count} outliers in column '{col}'. Capping them.")
                # Cap outliers
                df = df.with_columns(
                    pl.col(col).clip(lower_bound=lower_bound, upper_bound=upper_bound)
                )
        return df

    def final_validation(self, df: pl.DataFrame):
        """Performs final validation checks after cleaning."""
        print(f"\nFinal Data Info:\n{df.schema}")
        print(f"\nMissing Values After Cleaning:\n{df.select(pl.col('*').is_null().sum())}")
        print(f"\nDuplicate Rows After Cleaning: {df.is_duplicated().sum()}")
        print("\nData cleaning pipeline completed.")