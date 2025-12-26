import polars as pl
from polars import DataFrame

from scipy.stats import contingency


class Cramers:
    def __init__(self, df: DataFrame, cat1: str, cat2: str):
        self.df = df
        self.cat1 = cat1
        self.cat2 = cat2

    def compute_v(self):
        contigency_table = (
            self.df.group_by([self.cat1, self.cat2])
            .agg(pl.len().alias('count'))
            .to_pandas()
            .pivot(index=self.cat1, columns=self.cat2, values='count')
            .fillna(0)
            .astype(int)
        )

        return contingency.association(contigency_table, method='cramer')