"""Choose the reject threshold tau on tune data only, for one model or an ensemble (a.pt+b.pt+c.pt).

Per tau it reports, all on tune-side data:
  var_bal_acc / in_scope_reject   tune split (gold train speakers held out from training)
  oos_accept                      out-of-scope accepted as a command: the tune split's gold OOS clips
                                  plus the eval-only tune_oos pack (scripts/make_tune_oos.py)
  neg_misfire                     synthetic negatives of tune speakers (neg_tune)
and picks the smallest tau whose oos_accept <= --max-oos-accept.

  python scripts/tau_report.py --model exp/a/best.pt+exp/b/best.pt --pack data/packs --out results/tau_x.csv
"""
import argparse

import numpy as np
import pandas as pd

from vcm.eval import compute_metrics, decide, make_predictor, open_split, pick_device, predict_split
from vcm.labels import OOS_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", default="data/packs")
    ap.add_argument("--max-oos-accept", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    f = make_predictor(a.model, pick_device(a.device))
    tune = open_split(a.pack, "tune")
    t_c, t_s = predict_split(f, tune)
    o_c, _ = predict_split(f, open_split(a.pack, "tune_oos"))
    n_c, _ = predict_split(f, open_split(a.pack, "neg_tune"))
    gold_oos = tune.meta["cmd_idx"].to_numpy() == OOS_IDX
    oos_c = np.concatenate([t_c[gold_oos], o_c])
    rows = []
    for tau in np.round(np.arange(0, 0.96, 0.05), 2):
        m = compute_metrics(tune.meta, t_c, t_s, tau, detail=False)
        acc = lambda p: float((decide(p, np.zeros((len(p), 6, 3)), tau)[0] != OOS_IDX).mean())
        rows.append({"tau": tau, "var_bal_acc": m["variation_bal_acc"], "real_var_bal_acc": m["real_variation_bal_acc"],
                     "in_scope_reject": m["in_scope_false_reject"], "oos_accept": acc(oos_c),
                     "oos_accept_gold28": acc(t_c[gold_oos]), "oos_accept_tune_oos": acc(o_c),
                     "neg_misfire": acc(n_c)})
    df = pd.DataFrame(rows)
    print(f"model {a.model}\noos clips: {int(gold_oos.sum())} gold tune + {len(o_c)} tune_oos; neg_tune {len(n_c)}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ok = df[df.oos_accept <= a.max_oos_accept]
    if len(ok):
        print("chosen tau:", ok.iloc[0].to_dict())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
