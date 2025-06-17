import polars as pl
import logging

class DescriptiveStats:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def get_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate a summary of the DataFrame, including count, mean, min, max, etc.
        """
        try:
            self.logger.info("Generating DataFrame summary")
            return df.describe()
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise