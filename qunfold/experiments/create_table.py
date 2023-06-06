import argparse
import itertools
import numpy as np
import os
import pandas as pd

def main(
        input_path,
        output_path,
    ):
    print(f"Starting to generate a LaTeX table at {output_path} from {input_path}")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # read the results and find the winning package for each error_metric and method
    df = pd.read_csv(input_path, index_col=0)
    best_package = df \
        .groupby(["error_metric", "method"]) \
        .apply(lambda sdf: sdf["package"].to_numpy()[sdf["error"].argmin()]) \
        .rename("package") \
        .reset_index()

    # create a pivot table from the results
    pivot_rows = []
    for _, r in df.iterrows():
        pivot_row = {
            "index": r["method"],
            "column": f"{r['error_metric'].upper()} / {r['package']}",
            "value": f"{{{r['error']:.3f} \\pm {r['error_std']:.3f}}}",
        }
        if np.all(r["package"] == best_package[
                (best_package["error_metric"] == r["error_metric"]) &
                (best_package["method"] == r["method"])
                ]["package"]): # print each winner in boldface
            pivot_row["value"] = "\\mathbf" + pivot_row["value"]
        pivot_row["value"] = "$" + pivot_row["value"] + "$"
        pivot_rows.append(pivot_row)
    df = pd.DataFrame(pivot_rows)
    df = df.pivot_table(index="index", columns="column", values="value", aggfunc="first")
    df = df.reset_index(names="method").rename_axis(None, axis=1) # pandas crap

    # export a LaTeX table
    with open(output_path, "w") as f:
        print("\\begin{tabular}{l" + "c"*(len(df.columns)-1) + "}", file=f)
        print("  \\toprule", file=f)
        print("  " + " & ".join(df.columns) + " \\\\", file=f)
        print("  \\midrule", file=f)
        for _, r in df.iterrows():
            r[r.isna()] = "$\\ast$"
            print("  " + " & ".join(r) + " \\\\", file=f)
        print("  \\bottomrule", file=f)
        print("\\end{tabular}", file=f)
    print(f"LaTeX table succesfully stored at {output_path}:", df, sep="\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='path of an input *.csv file')
    parser.add_argument('output_path', type=str, help='path of an output *.tex file')
    args = parser.parse_args()
    main(
        args.input_path,
        args.output_path,
    )
