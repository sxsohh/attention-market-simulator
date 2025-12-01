import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")


def main():
    os.makedirs("visuals", exist_ok=True)
    data_path = Path("data") / "attention_simulation.csv"

    df = pd.read_csv(data_path)

    # time series plots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df["t"], df["attention_level"])
    axes[0].set_ylabel("Attention")

    axes[1].plot(df["t"], df["volatility"])
    axes[1].set_ylabel("Volatility")

    axes[2].plot(df["t"], df["attention_imbalance"])
    axes[2].set_ylabel("Imbalance")

    axes[3].plot(df["t"], df["boredom"], label="Boredom")
    axes[3].plot(df["t"], df["fatigue"], label="Fatigue")
    axes[3].set_ylabel("Boredom/Fatigue")
    axes[3].set_xlabel("Time step")
    axes[3].legend()

    fig.suptitle("Core State Variables Over Time", y=1.02)
    plt.tight_layout()
    fig.savefig("visuals/core_timeseries.png")
    plt.close(fig)

    # regime visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    regime_map = {
        "engaged": 0,
        "fatigued": 1,
        "overstimulated": 2,
        "addictive_loop": 3,
        "disengaged": 4,
        "baseline": 5,
    }
    scatter = ax.scatter(
        df["t"],
        df["attention_level"],
        c=df["regime"].map(regime_map),
        s=10,
        cmap="tab10",
    )
    ax.set_title("Attention Level with Regimes")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Attention level")
    plt.tight_layout()
    fig.savefig("visuals/regime_scatter_full.png")
    plt.close(fig)

    # return distributions (like in finance)
    df = df.copy()
    df["attn_return_1"] = df["attention_level"].diff()
    df["attn_abs_return_1"] = df["attn_return_1"].abs()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df["attn_return_1"].dropna(), bins=50)
    ax[0].set_title("Distribution of 1-step Attention Returns")
    ax[0].set_xlabel("Return")

    ax[1].hist(df["attn_abs_return_1"].dropna(), bins=50)
    ax[1].set_title("Distribution of |Return|")
    ax[1].set_xlabel("|Return|")

    plt.tight_layout()
    fig.savefig("visuals/returns_distributions.png")
    plt.close(fig)

    # does imbalance predict future attention?
    df["attn_fwd_change_5"] = df["attention_level"].shift(-5) - df["attention_level"]
    analysis_df = df.dropna(subset=["attn_fwd_change_5", "attention_imbalance"]).copy()

    sample = analysis_df.sample(min(1500, len(analysis_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=sample,
        x="attention_imbalance",
        y="attn_fwd_change_5",
        ax=ax,
    )
    ax.set_title("Attention Imbalance vs 5-step Forward Change")
    ax.set_xlabel("Current Imbalance")
    ax.set_ylabel("Future 5-step Change")
    plt.tight_layout()
    fig.savefig("visuals/imbalance_vs_future_change.png")
    plt.close(fig)

    # correlation heatmap
    feature_cols = [
        "attention_level",
        "volatility",
        "attention_imbalance",
        "attention_liquidity",
        "attention_demand",
        "boredom",
        "fatigue",
    ]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix of Core Features")
    plt.tight_layout()
    fig.savefig("visuals/core_correlation_heatmap.png")
    plt.close(fig)

    print("done! check visuals/ folder")


if __name__ == "__main__":
    main()
