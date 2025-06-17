import polars as pl
import numpy as np

class TrendAnalysis:
    def identify_time_series_trends(self, df: pl.DataFrame, time_column: str, value_column: str):
        """
        Identifies basic trends in time series data.
        Uses a simple moving average for smoothing and compares start/end values.
        """
        if time_column not in df.columns or value_column not in df.columns:
            return "Error: Specified time or value column not found."

        # Create a copy and clean the data
        df_copy = df.select([time_column, value_column]).clone()
        df_copy = df_copy.with_columns(
            pl.col(time_column).cast(pl.Datetime, strict=False)
        )
        df_copy = df_copy.filter(
            ~pl.col(time_column).is_null() & ~pl.col(value_column).is_null()
        )
        df_copy = df_copy.sort(time_column)

        if df_copy.height == 0:
            return "No valid time series data after cleaning."

        # Calculate a simple moving average (window size = 3)
        df_copy = df_copy.with_columns(
            pl.col(value_column).rolling_mean(window_size=3).alias("moving_avg")
        )

        # Get start and end values
        start_value = df_copy.select(pl.col(value_column).first()).item()
        end_value = df_copy.select(pl.col(value_column).last()).item()

        # Get time range
        time_min = df_copy.select(pl.col(time_column).min()).item()
        time_max = df_copy.select(pl.col(time_column).max()).item()

        trend_description = f"Over the period from {time_min.strftime('%Y-%m-%d')} to {time_max.strftime('%Y-%m-%d')}, "

        if end_value > start_value * 1.1:  # More than 10% increase
            trend_description += f"the {value_column} shows a significant upward trend, increasing from {start_value:.2f} to {end_value:.2f}."
        elif end_value < start_value * 0.9:  # More than 10% decrease
            trend_description += f"the {value_column} shows a significant downward trend, decreasing from {start_value:.2f} to {end_value:.2f}."
        else:
            trend_description += f"the {value_column} remained relatively stable, fluctuating between {start_value:.2f} and {end_value:.2f}."

        return trend_description

    def find_correlations(self, df: pl.DataFrame):
        """
        Calculates and describes correlations between numerical columns.
        """
        # Select numerical columns
        numeric_df = df.select(pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64))
        if len(numeric_df.columns) < 2:
            return "Need at least two numerical columns to find correlations."

        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        correlation_description = "Here are the correlations between numerical columns:\n"
        correlation_description += correlation_matrix.to_pandas().to_markdown() + "\n\n"

        # Describe strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_val = correlation_matrix[col1][j]
                if abs(corr_val) > 0.7:  # Threshold for strong correlation
                    relation = "strong positive" if corr_val > 0 else "strong negative"
                    strong_correlations.append(f"'{col1}' and '{col2}' show a {relation} correlation ({corr_val:.2f}).")
        
        if strong_correlations:
            correlation_description += "Notable correlations:\n" + "\n".join(strong_correlations)
        else:
            correlation_description += "No particularly strong correlations (absolute value > 0.7) found between numerical columns."
        
        return correlation_description