import polars as pl
import plotly.graph_objects as go

class PlotlyPlots:
    MAX_INLIERS = 200_000
    MAX_OUTLIERS = 3_000

    def __init__(self, df: pl.DataFrame):
        self.df = df

    def dist_plot(self, col: str, max_inliers: int = MAX_INLIERS,
                  max_outliers: int = MAX_OUTLIERS, seed: int = 42):
        s = self.df.select(pl.col(col).drop_nulls().cast(pl.Float64).alias("x"))["x"]

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lf = q1 - 1.5 * iqr
        uf = q3 + 1.5 * iqr

        df_x = self.df.select(pl.col(col).drop_nulls().cast(pl.Float64).alias("x"))

        inliers = (
            df_x.filter((pl.col("x") >= lf) & (pl.col("x") <= uf))
            .sample(n=min(max_inliers, df_x.height), seed=seed, with_replacement=False)
            .with_columns(pl.lit(False).alias("is_outlier"))
        )

        outliers = (
            df_x.filter((pl.col("x") < lf) | (pl.col("x") > uf))
            .with_columns(
                pl.when(pl.col("x") < lf).then(lf - pl.col("x")).otherwise(pl.col("x") - uf).alias("dist")
            )
            .sort("dist", descending=True)
            .head(max_outliers)
            .select(["x"])
            .with_columns(pl.lit(True).alias("is_outlier"))
        )

        plot_df = pl.concat([inliers, outliers], how="vertical_relaxed")
        d = plot_df.to_dict(as_series=False)

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=[v for v, o in zip(d["x"], d["is_outlier"]) if not o],
            name="Inliers (amostra)",
            nbinsx=60,
            histnorm="probability density",
            opacity=0.75,
        ))

        fig.add_trace(go.Histogram(
            x=[v for v, o in zip(d["x"], d["is_outlier"]) if o],
            name="Outliers (amostra)",
            nbinsx=60,
            histnorm="probability density",
            opacity=0.55,
        ))

        fig.update_layout(
            barmode="overlay",
            title=f"Distribuição - {col}",
            xaxis_title=col,
            yaxis_title="Densidade",
            legend_title_text="Grupo",
        )

        return fig