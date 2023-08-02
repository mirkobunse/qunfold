import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cond.csv", index_col=0)
df["color"] = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red"])[
    2 * df["is_probabilistic"].astype(int) + (df["class_weight"] == "balanced").astype(int)
]
descr = "blue=(ACC, unbal.), orange=(ACC, bal.), green=(PACC, unbal.), red=(PACC, bal.)"

metrics = ["ae", "rae", "se"]
conditions = ["cond_1", "cond_2", "cond_inf"]
# conditions = ["cond_1", "cond_-1", "cond_2", "cond_-2", "cond_inf", "cond_-inf", "cond_fro"]

for group_name, gdf in df.groupby("solver"):
    gdf = gdf.sort_values("C")
    fig, axs = plt.subplots(len(conditions), len(metrics), constrained_layout=True)
    fig.suptitle(f"solver={group_name} ({descr})")
    for i_metric, metric in enumerate(metrics):
        for i_cond, cond in enumerate(conditions):
            axs[i_cond,i_metric].set_title(f"{cond} - {metric}")
            axs[i_cond,i_metric].set_xscale("log")
            axs[i_cond,i_metric].set_yscale("log")
            axs[i_cond,i_metric].set_xlim(None, 1e9)
            for color_name, cdf in gdf.groupby("color"):
                axs[i_cond,i_metric].plot(cdf[cond], cdf[metric], c=color_name)
                if color_name == "tab:red":
                    axs[i_cond,i_metric].text(list(cdf[cond])[0], list(cdf[metric])[0], f"C={list(cdf['C'])[0]}", c=color_name)
                    axs[i_cond,i_metric].text(list(cdf[cond])[-1], list(cdf[metric])[-1], f"C={list(cdf['C'])[-1]}", c=color_name)
    plt.show()
