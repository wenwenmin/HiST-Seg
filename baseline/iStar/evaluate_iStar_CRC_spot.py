
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


SLICES=[
"SN048_1","SN048_2","SN048_3","SN048_4",
"SN84_1","SN84_2",
"SN123_1","SN123_2","SN123_4",
"SN124_1","SN124_2","SN124_4"
]

PRED_ROOT=Path(
r"D:\DNM_project(cancer detect)\predict_files\iStar\CRC_spot_level"
)

LABEL_ROOT=Path(
r"D:\DNM_project(cancer detect)\labels\CRC"
)

OUT=Path("./iStar_CRC_metrics")
OUT.mkdir(exist_ok=True)

rows=[]

for s in SLICES:
    y_raw=load_txt(
        LABEL_ROOT/f"{s}-labels3.txt",
        int
    )

    valid=y_raw!=-2
    y=np.where(y_raw[valid]==2,1,0)

    for seed in SEEDS:
        prob=load_txt(
            PRED_ROOT/f"{s}_probs_seed{seed}.txt",
            float
        )

        if len(prob)==len(valid):
            prob=prob[valid]

        r=metrics(y,prob)
        r.update(
            Slice=s,
            Seed=seed,
            Spots=len(y)
        )
        rows.append(r)

save_summary(rows,SLICES,OUT)
