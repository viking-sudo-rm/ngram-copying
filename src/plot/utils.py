from math import log, floor
import matplotlib.ticker
import seaborn as sns
import matplotlib.pyplot as plt  # Technically unused.

from .constants import *

def clean_model_name(model):
    return model.replace("EleutherAI/", "")

def clean_size(size) -> float:
    if size.endswith("m"):
        return float(size[:-1]) * 1e6
    elif size.endswith("b"):
        return float(size[:-1]) * 1e9
    else:
        raise ValueError

def clean_size_string(key) -> str:
    return key.split("-")[-1].replace("m", "M").replace("b", "B")

def format_novelty_plot(args, plt):
    plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.xlabel(R"$n$-gram size")
    plt.ylabel("% novel")
    plt.ylim([0, 1])
    if args.log_scale:
        plt.xscale("log")
        max_e = floor(log(args.max_n + 1, 10))
        ticks = [10**e for e in range(0, max_e + 1)]
        plt.xticks(ticks)
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        plt.xticks(list(range(1, args.max_n + 1)))
    plt.tight_layout()
    sns.despine()  # Not called on passed object.

def plot_baselines(plt, nov, val, val_dolma, show_lb=False):
    if show_lb:
        sizes, lb_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
        plt.fill_between(sizes, 0, lb_unique, color=MUTED_RED, alpha=0.2, label="LB")
        # plt.plot(sizes, lb_unique, linestyle="-", color=MUTED_RED)

    sizes, red_unique = nov.get_proportion_unique(val_dolma)
    plt.plot(sizes, red_unique, label="Dolma", color=GRAY, linestyle="--")

    sizes, val_unique = nov.get_proportion_unique(val)
    plt.plot(sizes, val_unique, label="Valid", color=LIGHT_GRAY, linestyle="-.")

    if show_lb:
        return val_unique, red_unique, lb_unique
    else:
        return val_unique, red_unique