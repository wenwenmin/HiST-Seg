# CGS-Net on STHBC LOSO

本文件夹用于把原始 CGS-Net 项目适配到你的 8 张 STHBC 切片癌症区域像素级分割任务。

## 1. 放置位置

把整个 `sthbc_loso` 文件夹复制到：

```text
D:/Project_2/Pixel_Seg/baseline/CGS-Net/sthbc_loso
```

最终结构应类似：

```text
D:/Project_2/Pixel_Seg/baseline/CGS-Net/
├── dlp/
├── weights/
│   └── single_mit_b1_imagenet.pth
└── sthbc_loso/
    ├── sthbc_loso.yaml
    ├── cgs_sthbc_lib.py
    ├── train_sthbc_loso.py
    └── inference_sthbc_loso.py
```

## 2. 数据路径

默认配置使用：

```text
D:/Project_2/Pixel_Seg/data/STHBC/resized_data/{slice}/he_resized.png
D:/Project_2/Pixel_Seg/data/STHBC/resized_data/{slice}/label_resized.png
D:/Project_2/Pixel_Seg/data/STHBC/resized_data/{slice}/mask_resized.png
```

切片顺序：

```text
C1, D1, H1, E1, F1, G2, A1, B1
```

## 3. 先检查数据完整性

在 `D:/Project_2/Pixel_Seg/baseline/CGS-Net` 根目录运行：

```bash
python sthbc_loso/train_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --check-only
```

会输出：

```text
results_sthbc_loso/dataset_check/dataset_check.csv
```

## 4. 训练一个fold

例如测试切片为 C1：

```bash
python sthbc_loso/train_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --fold C1
```

该 fold 自动执行：

1. 生成 train/val patch manifest；
2. 训练 detail 单分支；
3. 训练 context 单分支；
4. 用两个单分支初始化 CGS-Net；
5. 先冻结两个 encoder 与 lv2 decoder，仅训练 MCA；
6. 解冻全模型；
7. 用完整验证切片选择最佳 `cgs_best.pth`；
8. 对测试切片整图推理并保存结果。

## 5. 训练全部 8 折

```bash
python sthbc_loso/train_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --fold all
```

## 6. 重新推理全部fold

训练完成后，如果只想重新用已保存的 checkpoint 推理：

```bash
python sthbc_loso/inference_sthbc_loso.py --config sthbc_loso/sthbc_loso.yaml --fold all
```

## 7. 输出文件

每折位于：

```text
results_sthbc_loso/fold_{test_slice}/
```

主要文件：

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

总表：

```text
results_sthbc_loso/sthbc_loso_test_summary.csv
results_sthbc_loso/sthbc_loso_mean_std.csv
```

## 8. 默认fold划分

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

## 9. 调参建议

默认 epoch 是为了先跑通流程，不是论文中的 1000/600 epoch。最终实验可在 `sthbc_loso.yaml` 中提高：

```yaml
detail_epochs: 80
context_epochs: 80
cgs_epochs: 120
```

若显存不足，优先调小：

```yaml
batch_size: 4
eval_batch_size: 8
```

若整图推理慢，增大：

```yaml
inference:
  stride: 168
```

若希望更平滑但更慢，减小：

```yaml
inference:
  stride: 56
```
