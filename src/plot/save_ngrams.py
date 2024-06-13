import os

class SaveNgrams:
    """Save n-grams in model-generated text for qualitative analysis."""

    def save(self, lengths):
        plots_dir = os.path.join(PLOTS, "completion-loss")
        os.makedirs(plots_dir, exist_ok=True)