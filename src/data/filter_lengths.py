from collections import defaultdict

from src.plot import clean_model_name

class FilterLengths:
    def __init__(self, lmg, results):
        self.lmg = lmg
        self.results = results

    def get_baselines(self, key="lengths") -> dict:
        lengths = {}
        lengths["val"] = self.results.val_iid[key]
        lengths["val_reddit"] = self.results.val_reddit[key]
        lengths["val_cc"] = self.results.val_cc[key]
        lengths["val_stack"] = self.results.val_stack[key]
        lengths["val_pes2o"] = self.results.val_pes2o[key]
        return lengths


    def get_lengths_for_model(self, model: str, key="lengths") -> dict:
        plot_lengths = defaultdict(list)
        plot_lengths.update(**self.get_baselines(key))
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
        plot_lengths.update(**self.get_baselines())
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