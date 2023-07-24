from river.evaluate import progressive_val_score


def evaluate(dataset, model, rep, metric, n_samples, approach_name, ds_name: str = None):
    score = progressive_val_score(dataset.take(n_samples), model, metric)
    return [dataset.__class__.__name__ if ds_name is None else ds_name,
            approach_name,
            rep,
            score.get(),
            metric.__class__.__name__]