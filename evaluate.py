from river.evaluate import progressive_val_score


def evaluate(dataset, model, rep, metric, n_samples, approach_name):
    score = progressive_val_score(dataset.take(n_samples), model, metric)
    return [dataset.__class__.__name__,
            approach_name,
            rep,
            score.get(),
            metric.__class__.__name__]