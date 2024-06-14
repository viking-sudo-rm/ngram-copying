import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from . import *
from ..data import LMGenerations, Results
from ..ngram_novelty import NgramNovelty

# Decoding parameters, from most to least constrained.
KS = [20, 80, 160, "$\\infty$"]
PS = [0.85, 0.9, 0.95, 1.0]
TEMPS = [0., 0.5, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
BEAMS = [8, 4, 1]

class DecodingPlots:
    
    def __init__(self, args):
        self.args = args

    def plot_by_topp(self, lmg: LMGenerations, results: Results, lengths_12b: dict):
        print("\n=== plot_by_topp ===")
        os.makedirs("plots/by-decoding", exist_ok=True)
        nov = NgramNovelty(self.args.max_n)

        lengths_by_topp = defaultdict(list)
        for doc, lengths in zip(lmg.by_topp, results.by_topp["lengths"]):
            decoding = doc["meta"]["decoding"]
            plen = doc["meta"]["prompt_len"]
            lengths_by_topp[plen, decoding].append(lengths)
        for plen in PLENS:
            lengths_by_topp[plen, "top_p=1.0"] = lengths_12b[plen]
    
        # FIXME: p is overloaded between prefix length and here.
        for plen in PLENS:
            plt.figure()
            lengths_dolma = lengths_12b["val_reddit"] + lengths_12b["val_pes2o"]
            plot_baselines(plt, nov, lengths_12b["val"], lengths_dolma)
            
            colors = plt.cm.Purples(np.linspace(0.2, 1.0, len(PS)))
            for color, p in zip(colors, reversed(PS)):
                lengths = lengths_by_topp[plen, f"top_p={p}"]
                sizes, prop_unique = nov.get_proportion_unique(lengths)
                plt.plot(sizes, prop_unique, label=Rf"$p$={p}", color=color)

            title = R"$n$-novelty by top-$p$"
            if not self.args.no_plen:
                title += f" (plen={plen})"
            plt.title(title)
            format_novelty_plot(self.args, plt)
            plt.savefig(f"plots/by-decoding/topp-{plen}.pdf")
            plt.close()

    def plot_by_topk(self, lmg: LMGenerations, results: Results, lengths_12b: dict):
        print("\n=== plot_by_topk ===")
        os.makedirs("plots/by-decoding", exist_ok=True)
        nov = NgramNovelty(self.args.max_n)

        lengths_by_topk = defaultdict(list)
        for doc, lengths in zip(lmg.by_topk, results.by_topk["lengths"]):
            decoding = doc["meta"]["decoding"]
            plen = doc["meta"]["prompt_len"]
            lengths_by_topk[plen, decoding].append(lengths)
        for plen in PLENS:
            lengths_by_topk[plen, "top_k=$\\infty$"] = lengths_12b[plen]
    
        for plen in PLENS:
            plt.figure()
            lengths_dolma = lengths_12b["val_reddit"] + lengths_12b["val_pes2o"]
            plot_baselines(plt, nov, lengths_12b["val"], lengths_dolma)
            
            colors = plt.cm.Purples(np.linspace(0.2, 1.0, len(KS)))
            for color, k in zip(colors, reversed(KS)):
                lengths = lengths_by_topk[plen, f"top_k={k}"]
                sizes, prop_unique = nov.get_proportion_unique(lengths)
                plt.plot(sizes, prop_unique, label=Rf"$k$={k}", color=color)

            title = R"$n$-novelty by top-$k$"
            if not self.args.no_plen:
                title += f" (plen={plen})"
            plt.title(title)
            format_novelty_plot(self.args, plt)
            plt.savefig(f"plots/by-decoding/topk-{plen}.pdf")
            plt.close()

    def plot_by_temp(self, lmg: LMGenerations, results: Results, lengths_12b: dict):
        print("\n=== plot_by_temp ===")
        os.makedirs("plots/by-decoding", exist_ok=True)
        nov = NgramNovelty(self.args.max_n)

        lengths_by_temp = defaultdict(list)
        for doc, lengths in zip(lmg.by_temp, results.by_temp["lengths"]):
            decoding = doc["meta"]["decoding"]
            plen = doc["meta"]["prompt_len"]
            lengths_by_temp[plen, decoding].append(lengths)
        for plen in PLENS:
            lengths_by_temp[plen, "temp=1.0"] = lengths_12b[plen]
        for doc, lengths in zip(lmg.beam1, results.beam1["lengths"]):
            plen = doc["meta"]["prompt_len"]
            lengths_by_temp[plen, "temp=0.0"].append(lengths)
    
        # Only ran temperature experiments with plen=0.
        for plen in [0]:
            plt.figure()
            lengths_dolma = lengths_12b["val_reddit"] + lengths_12b["val_pes2o"]
            plot_baselines(plt, nov, lengths_12b["val"], lengths_dolma)

            colors = plt.cm.Purples(np.linspace(0.2, 1.0, len(TEMPS)))
            for color, temp in zip(colors, reversed(TEMPS)):
                lengths = lengths_by_temp[plen, f"temp={temp}"]
                sizes, prop_unique = nov.get_proportion_unique(lengths)
                plt.plot(sizes, prop_unique, label=Rf"$\tau$={temp}", color=color)
                n10 = prop_unique[sizes == 10].item()
                n100 = prop_unique[sizes == 100].item()
                print(f"temp={temp}: {n10:.2f} @ 10, {n100:.2f} @ 100")

            title = R"$n$-novelty by temperature"
            if not self.args.no_plen:
                title += f" (plen={plen})"
            plt.title(title)
            format_novelty_plot(self.args, plt)
            plt.savefig(f"plots/by-decoding/temp-{plen}.pdf")
            plt.close()

    def plot_by_beam(self, lmg: LMGenerations, results: Results, lengths_12b: dict):
        print("\n=== plot_by_beam ===")
        os.makedirs("plots/by-decoding", exist_ok=True)
        nov = NgramNovelty(self.args.max_n)

        lengths_by_beam = defaultdict(list)
        for doc, lengths in zip(lmg.beam1, results.beam1["lengths"]):
            plen = doc["meta"]["prompt_len"]
            lengths_by_beam[plen, "beam=1"].append(lengths)
        for doc, lengths in zip(lmg.beam4, results.beam4["lengths"]):
            plen = doc["meta"]["prompt_len"]
            lengths_by_beam[plen, "beam=4"].append(lengths)
        for doc, lengths in zip(lmg.beam8, results.beam8["lengths"]):
            plen = doc["meta"]["prompt_len"]
            lengths_by_beam[plen, "beam=8"].append(lengths)
    
        for plen in PLENS:
            plt.figure()
            lengths_dolma = lengths_12b["val_reddit"] + lengths_12b["val_pes2o"]
            plot_baselines(plt, nov, lengths_12b["val"], lengths_dolma)
            
            colors = plt.cm.Purples(np.linspace(0.2, 1.0, len(BEAMS)))
            for color, beam in zip(colors, reversed(BEAMS)):
                lengths = lengths_by_beam[plen, f"beam={beam}"]
                sizes, prop_unique = nov.get_proportion_unique(lengths)
                plt.plot(sizes, prop_unique, label=Rf"$b$={beam}", color=color)
                n10 = prop_unique[sizes == 10].item()
                n100 = prop_unique[sizes == 100].item()
                print(f"beam={beam}: {n10:.2f} @ 10, {n100:.2f} @ 100")

            title = R"$n$-novelty by beam size"
            if not self.args.no_plen:
                title += f" (plen={plen})"
            plt.title(title)
            format_novelty_plot(self.args, plt)
            plt.savefig(f"plots/by-decoding/beam-{plen}.pdf")
            plt.close()