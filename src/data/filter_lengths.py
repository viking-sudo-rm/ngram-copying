from collections import defaultdict

from src.plots import clean_model_name

class FilterLengths:
    def __init__(self, lmg, results):
        self.lmg = lmg
        self.results = results

    def get_lengths_for_model(self, model: str, key="lengths") -> dict:
        plot_lengths = defaultdict(list)
        plot_lengths["val"] = self.results.val_iid[key]
        plot_lengths["val_reddit"] = self.results.val_reddit["lengths"]
        for doc, lengths in zip(self.lmg.by_model, self.results.by_model[key]):
            if doc["meta"]["model"] != model:
                continue
            plen = doc["meta"]["prompt_len"]
            plot_lengths[plen].append(lengths)
        for doc, lengths in zip(self.lmg.by_model_p, self.results.by_model_p[key]):
            if doc["meta"]["model"] != model:
                continue
            plen = doc["meta"]["prompt_len"]
            plot_lengths[plen].append(lengths)
        return plot_lengths

    def get_by_model(self, filter_fn=None):
        plot_lengths = defaultdict(list)
        plot_lengths["val"] = self.results.val_iid["lengths"]
        plot_lengths["val_reddit"] = self.results.val_reddit["lengths"]
        for doc, lengths in zip(self.lmg.by_model, self.results.by_model["lengths"]):
            if filter_fn and not filter_fn(doc["meta"]):
                continue
            model = clean_model_name(doc["meta"]["model"])
            plot_lengths[model].append(lengths)
        return plot_lengths

    def get_by_decoding(self, field, filter_fn=None):
        dec_lengths = defaultdict(list)
        lmg = getattr(self.lmg, field)
        results = getattr(self.results, field)
        for doc, lengths in zip(lmg, results["lengths"]):
            if filter_fn and not filter_fn(doc["meta"]):
                continue
            decoding = doc["meta"]["decoding"]
            dec_lengths[decoding].append(lengths)
        return dec_lengths