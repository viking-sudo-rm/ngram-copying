import numpy as np
from collections import defaultdict

from ..data import Results, LMGenerations
from ..data.utils import flatten
from ..data.filter_lengths import FilterLengths

def get_entries(lengths):
    lengths = flatten(lengths)
    return {
        "median": np.median(lengths),
        "mean": np.mean(lengths),
        "max": np.max(lengths),
    }

def fmt_cell(x, fmt="float"):
    if fmt == "float":
        return x if isinstance(x, str) else f"{x:.2f}"
    else:
        return x if isinstance(x, str) else f"{int(x):d}"

MODEL_SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
PLENS = [1, 10, 100]

# TODO: Refactor out Table class that has add_column(), generate_rows(), write_tex(fh)

class SaveTable:
    def save(self, lmg: LMGenerations, results: Results):
        filter_fn = lambda meta: meta["prompt_len"] == 0
        flengths = FilterLengths(lmg, results)
        by_model = flengths.get_by_model(filter_fn)
        by_topp = flengths.get_by_decoding("by_topp", filter_fn)
        by_topk = flengths.get_by_decoding("by_topk", filter_fn)
        by_temp = flengths.get_by_decoding("by_temp", filter_fn)
        by_beam = flengths.get_by_decoding("beam1", filter_fn)  # temperature 0
        by_beam.update(flengths.get_by_decoding("beam4", filter_fn))
        by_beam.update(flengths.get_by_decoding("beam8", filter_fn))
        by_plen = flengths.get_lengths_for_model("EleutherAI/pythia-12b")

        ##########
        # Table 1.
        ##########

        cols1 = {}
        cols1["control", "val"] = get_entries(by_model["val"])
        cols1["control", "reddit"] = get_entries(by_model["val_reddit"])
        for size in MODEL_SIZES:
            data = by_model["pythia-" + size]
            cols1["model size", size] = get_entries(data)
        for plen in PLENS:
            data = by_plen[plen]
            cols1["plen", str(int(plen))] = get_entries(data)

        rows1 = [
            [""], [""],
            ["Med"], ["Mean"], ["Max"],
        ]
        for (cat, name), col in cols1.items():
            rows1[0].append(cat)
            rows1[1].append(name)
            rows1[2].append(col["median"])
            rows1[3].append(col["mean"])
            rows1[4].append(col["max"])

        ##########
        # Table 2.
        ##########

        cols2 = {}
        cols2["control", "val"] = get_entries(by_model["val"])
        cols2["control", "reddit"] = get_entries(by_model["val_reddit"])
        for decoding_data in [by_topp, by_topk, by_temp, by_beam]:
            for name, data in decoding_data.items():
                dec, value = name.split("=")
                cols2[dec, value] = get_entries(data)

        rows2 = [
            [""], [""],
            ["Med"], ["Mean"], ["Max"],
        ]
        for (cat, name), col in cols2.items():
            rows2[0].append(cat)
            rows2[1].append(name)
            rows2[2].append(col["median"])
            rows2[3].append(col["mean"])
            rows2[4].append(col["max"])

        ##########
        # Render!
        ##########

        with open("tables/suffix-lengths.tex", "w") as fh:
            fh.write(R"\begin{tabular}{l|" + "c" * len(cols1) + "}\n")
            fh.write("    \\toprule\n")
            # Header
            # fh.write("    " + " & ".join(rows[0]) + R" \\" + "\n")
            # Title
            fh.write("    " + " & ".join([fmt_cell(x) for x in rows1[1]]) + R" \\" + "\n")
            fh.write("    \\hline\n")
            # Data
            fh.write("    " + " & ".join([fmt_cell(x, fmt="int") for x in rows1[2]]) + R" \\" + "\n")
            fh.write("    " + " & ".join([fmt_cell(x, fmt="float") for x in rows1[3]]) + R" \\" + "\n")
            fh.write("    " + " & ".join([fmt_cell(x, fmt="int") for x in rows1[4]]) + R" \\" + "\n")
            fh.write("    \\bottomrule\n")
            fh.write(R"\end{tabular}" + "\n")
            fh.write("\n")

            fh.write(R"\begin{tabular}{l|" + "c" * len(cols2) + "}\n")
            fh.write("    \\toprule\n")
            # Header
            # fh.write("    " + " & ".join(rows[0]) + R" \\" + "\n")
            # Title
            fh.write("    " + " & ".join([fmt_cell(x) for x in rows2[1]]) + R" \\" + "\n")
            fh.write("    \\hline\n")
            # Data
            fh.write("    " + " & ".join([fmt_cell(x, fmt="int") for x in rows2[2]]) + R" \\" + "\n")
            fh.write("    " + " & ".join([fmt_cell(x, fmt="float") for x in rows2[3]]) + R" \\" + "\n")
            fh.write("    " + " & ".join([fmt_cell(x, fmt="int") for x in rows2[4]]) + R" \\" + "\n")
            fh.write("    \\bottomrule\n")
            fh.write(R"\end{tabular}" + "\n")