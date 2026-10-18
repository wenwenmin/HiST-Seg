
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, accuracy_score
)

SEEDS = [42,123,567]
THRESHOLD = 0.5


def load_txt(path, dtype):
    x = np.loadtxt(path, dtype=dtype)
    return np.asarray(x).reshape(-1)


def metrics(y_true, prob):
    pred = (prob >= THRESHOLD).astype(int)

    out = {
        "AUC": roc_auc_score(y_true, prob),
        "AP": average_precision_score(y_true, prob),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
    }

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0,1]
    ).ravel()

    out["Sensitivity"] = tp/(tp+fn) if tp+fn else 0
    out["Specificity"] = tn/(tn+fp) if tn+fp else 0
    return out


def save_summary(rows, slices, out):
    df = pd.DataFrame(rows)
    df.to_csv(
        out/"per_seed_metrics.csv",
        index=False
    )

    summary=[]
    for s in slices:
        tmp=df[df.Slice==s]
        row={"Slice":s}
        for m in [
            "AUC","AP","F1",
            "Accuracy",
            "Sensitivity",
            "Specificity"
        ]:
            row[m]=(
                f"{tmp[m].mean():.4f}"
                f" ± {tmp[m].std():.4f}"
            )
        summary.append(row)

    pd.DataFrame(summary).to_csv(
        out/"summary_mean_std.csv",
        index=False
    )

    print(pd.DataFrame(summary).to_string(index=False))


SLICES=["ViHBC","DuCIS","XeHBC"]

ROOT=Path(
r"D:\DNM_project(cancer detect)\predict_files\iStar"
)

LABELS={
"ViHBC":Path("./labels/ViHBC.txt"),
"DuCIS":Path("./labels/DuCIS.txt"),
"XeHBC":Path("./labels/XeHBC.txt"),
}

OUT=Path("./iStar_VDX_metrics")
OUT.mkdir(exist_ok=True)

rows=[]

for s in SLICES:
    y=load_txt(LABELS[s],int)

    for seed in SEEDS:
        prob=load_txt(
            ROOT/f"{s}_spot_level"/
            f"probs_seed{seed}.txt",
            float
        )

        r=metrics(y,prob)
        r.update(
            Slice=s,
            Seed=seed,
            Spots=len(y)
        )
        rows.append(r)

save_summary(rows,SLICES,OUT)
