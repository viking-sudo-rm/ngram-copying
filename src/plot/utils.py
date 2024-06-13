from math import log, floor
import matplotlib.ticker
import seaborn as sns

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

def plot_baselines(plt, nov, val, val_reddit, show_lb=True):
    sizes, red_unique = nov.get_proportion_unique(val_reddit)
    plt.plot(sizes, red_unique, label="Reddit", color=GRAY, linestyle="--")

    sizes, val_unique = nov.get_proportion_unique(val)
    plt.plot(sizes, val_unique, label="val", color=LIGHT_GRAY, linestyle="--")

    if show_lb:
        sizes, lb_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
        plt.plot(sizes, lb_unique, linestyle="--", color=MUTED_RED, label="LB")
        return val_unique, red_unique, lb_unique
    else:
        return val_unique, red_unique