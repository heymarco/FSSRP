from river.evaluate import progressive_val_score
from river.metrics import Accuracy


def evaluate(dataset, model, rep, metric, n_samples):
    score = progressive_val_score(dataset.take(n_samples), model, metric)
    return [dataset.__class__.__name__,
            model.__class__.__name__,
            rep,
            score.get(),
            metric.__class__.__name__]