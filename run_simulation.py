from src.environment import AttentionEnv
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # make sure output dirs exist
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # run the simulation
    env = AttentionEnv(max_time_steps=2000)
    history = env.run()

    df = pd.DataFrame(history)
    print(df.head())
    print(df["regime"].value_counts())

    # save for analysis
    df.to_csv("data/attention_simulation.csv", index=False)
    print("\nSaved to data/attention_simulation.csv\n")

    # make some quick plots
    # attention over time
    plt.figure(figsize=(12,4))
    plt.plot(df["t"], df["attention_level"])
    plt.title("Attention Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Attention Level")
    plt.tight_layout()
    plt.savefig("visuals/attention_over_time.png")
    plt.show()

    # imbalance over time
    plt.figure(figsize=(12,4))
    plt.plot(df["t"], df["attention_imbalance"])
    plt.title("Attention Imbalance")
    plt.xlabel("Time Step")
    plt.ylabel("Imbalance")
    plt.tight_layout()
    plt.savefig("visuals/imbalance_over_time.png")
    plt.show()

    # color by regime
    plt.figure(figsize=(12,4))
    regime_map = {
        "engaged": 0,
        "fatigued": 1,
        "overstimulated": 2,
        "addictive_loop": 3,
        "disengaged": 4,
        "baseline": 5
    }
    plt.scatter(df["t"], df["attention_level"], 
                c=df["regime"].map(regime_map), cmap="tab10", s=10)
    plt.title("Attention Regimes Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Attention")
    plt.tight_layout()
    plt.savefig("visuals/regime_scatter.png")
    plt.show()


if __name__ == "__main__":
    main()
