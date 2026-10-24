# CGS-Net on STHBC LOSO

Replace the example path for your environment; see readme_baseline.txt.


Replace the example path for your environment; see readme_baseline.txt.

```text
./sthbc_loso
```


```text
./
├── dlp/
├── weights/
│   └── single_mit_b1_imagenet.pth
└── sthbc_loso/
    ├── sthbc_loso.yaml
    ├── cgs_sthbc_lib.py
    ├── train_sthbc_loso.py
    └── inference_sthbc_loso.py
```

# Replace the example path for your environment; see readme_baseline.txt.


```text
data/STHBC/resized_data/{slice}/he_resized.png
data/STHBC/resized_data/{slice}/label_resized.png
data/STHBC/resized_data/{slice}/mask_resized.png
```


```text
C1, D1, H1, E1, F1, G2, A1, B1
```


Run the command from the corresponding baseline directory.

```bash
python sthbc_loso/train_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --check-only
```


```text
results_sthbc_loso/dataset_check/dataset_check.csv
```



```bash
python sthbc_loso/train_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --fold C1
```




```bash
python sthbc_loso/train_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --fold all
```



```bash
python sthbc_loso/inference_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --fold all
```



```text
results_sthbc_loso/fold_{test_slice}/
```


```text
manifests/train_manifest.csv
manifests/val_manifest.csv
single_detail/single_detail_best.pth
single_context/single_context_best.pth
cgs/cgs_best.pth
cgs/cgs_log.csv
cgs/test/{slice}_prob.npy
cgs/test/{slice}_prob_heatmap.png
cgs/test/{slice}_pred_mask.png
cgs/test/{slice}_label_overlay.png
cgs/test/{slice}_pred_overlay.png
cgs/test/{slice}_prediction_only_overlay.png
cgs/test/{slice}_metrics.json
```


```text
results_sthbc_loso/sthbc_loso_test_summary.csv
results_sthbc_loso/sthbc_loso_mean_std.csv
```


| Fold/Test | Val | Train |
|---|---|---|
| C1 | D1 | H1,E1,F1,G2,A1,B1 |
| D1 | H1 | C1,E1,F1,G2,A1,B1 |
| H1 | E1 | C1,D1,F1,G2,A1,B1 |
| E1 | F1 | C1,D1,H1,G2,A1,B1 |
| F1 | G2 | C1,D1,H1,E1,A1,B1 |
| G2 | A1 | C1,D1,H1,E1,F1,B1 |
| A1 | B1 | C1,D1,H1,E1,F1,G2 |
| B1 | C1 | D1,H1,E1,F1,G2,A1 |



```yaml
detail_epochs: 80
context_epochs: 80
cgs_epochs: 120
```


```yaml
batch_size: 4
eval_batch_size: 8
```


```yaml
inference:
  stride: 168
```


```yaml
inference:
  stride: 56
```
