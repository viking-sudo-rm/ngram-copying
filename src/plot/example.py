from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import format_novelty_plot
from ..ngram_novelty import NgramNovelty

# Corpus: hello world
STRINGS = {
    "zebra": [0, 1, 0, 1, 0],
    "lloyd": [1, 2, 3, 0, 1],
    "cello": [0, 1, 2, 3, 4],
}

GREEN = (0, 0, 139 / 255)
RED = (139 / 255, 0, 0)
CM = LinearSegmentedColormap.from_list("green_red", [GREEN, RED], N=3)
COLORS = [CM(i / (len(STRINGS) - 1)) for i in range(len(STRINGS))]

class ExamplePlots:
    def __init__(self, args):
        self.args = deepcopy(args)
        self.args.log_scale = False
        self.args.max_n = 5
    
    def plot_example(self):
        nov = NgramNovelty(self.args.max_n)

        plt.figure()
        for color, (string, lengths) in zip(COLORS, STRINGS.items()):
            sizes, prop_unique = nov.get_proportion_unique([lengths])
            plt.plot(sizes, prop_unique, label=string, color=color)
        format_novelty_plot(self.args, plt)
        plt.legend(title=R"$C$: hello\$world\$", loc="lower right")
        plt.savefig("plots/example.pdf")
        plt.close()