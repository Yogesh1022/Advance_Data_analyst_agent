import polars as pl
import matplotlib.pyplot as plt
import logging

class ChartGenerator:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def generate_chart(self, query: str, df: pl.DataFrame) -> dict:
        """
        Generate a chart based on the query and DataFrame.
        Placeholder implementation for a simple bar chart.
        """
        try:
            self.logger.info(f"Generating chart for query: {query}")
            if "bar" in query.lower():
                # Example: Plot first numerical column
                num_col = df.select(pl.col(pl.Int64, pl.Float64)).columns[0]
                plt.figure()
                plt.bar(df[num_col].to_pandas(), height=df[num_col].to_pandas())
                plt.xlabel(num_col)
                plt.ylabel("Value")
                plt.title(f"Bar Chart of {num_col}")
                return {"plot_type": "matplotlib", "figure": plt.gcf(), "description": f"Bar chart of {num_col}"}
            return "Chart type not supported yet."
        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}")
            return f"Error generating chart: {str(e)}"