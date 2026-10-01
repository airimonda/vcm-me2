from vcm.early_stop import EarlyStopper


def run(losses, scores, **kw):
    es = EarlyStopper(**kw)
    for i, (l, s) in enumerate(zip(losses, scores), 1):
        if es.update(l, s):
            return i, es
    return None, es


def test_fires_only_when_both_conditions_hold():
    # loss plateau (flat) AND score falling after a peak -> stop
    losses = [1.0 - 0.1 * i for i in range(8)] + [0.30] * 10
    scores = [0.1 * i for i in range(8)] + [0.7, 0.69, 0.68, 0.67, 0.66, 0.65, 0.6, 0.5, 0.4, 0.3]
    stop, es = run(losses, scores, min_epochs=8)
    assert stop is not None and "early_stop" in es.reason
    # at the stop epoch both conditions really hold
    t = stop - 1
    assert (es.losses[t - 3] - es.losses[t]) / es.losses[t - 3] < 0.02
    assert es.scores[t] < es.best_score and es.scores[t] < es.scores[t - 3]


def test_no_stop_when_only_loss_plateaus():
    losses = [1.0 / (i + 1) for i in range(5)] + [0.2] * 30          # plateau
    scores = [0.5 + 0.01 * i for i in range(35)]                      # test keeps improving
    stop, es = run(losses, scores, max_epochs=35)
    assert es.reason.startswith("max_epochs")


def test_no_stop_when_only_score_drops():
    losses = [2.0 * 0.9 ** i for i in range(40)]                      # keeps improving >2%/3 epochs
    scores = [0.8] + [0.8 - 0.01 * i for i in range(1, 40)]
    stop, es = run(losses, scores, max_epochs=40)
    assert es.reason.startswith("max_epochs")


def test_min_epochs_guard():
    losses = [1.0] * 20
    scores = [0.9] + [0.5 - 0.01 * i for i in range(19)]
    stop, _ = run(losses, scores, min_epochs=8)
    assert stop == 8                                                  # earliest allowed


def test_max_epochs_guard():
    losses = [1.0 * 0.5 ** i for i in range(100)]
    scores = [0.01 * i for i in range(100)]
    stop, es = run(losses, scores, max_epochs=60)
    assert stop == 60 and es.reason.startswith("max_epochs")


def test_best_epoch_and_state_roundtrip():
    es = EarlyStopper()
    for l, s in [(1, .1), (.9, .3), (.8, .2)]:
        es.update(l, s)
    assert es.best_epoch == 2
    es2 = EarlyStopper()
    es2.load_state_dict(es.state_dict())
    assert es2.best_epoch == 2 and es2.losses == es.losses
