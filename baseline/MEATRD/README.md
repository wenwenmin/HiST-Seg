
## Usage
Please download the processed dataset and store it in the ```/data/``` directory, then run:
```
python main.py --data_path ./data/
```

## Code organization
- `configs/` - Configuration files for each task and dataset  --**没找到**
- `data/` - The dataset is pre-processed by DGL  --**存E盘即可**
- `utils.py` - Util functions (including evaluation suite)  --**随机种子和评估，可取消**
- `main.py` - Main python script for running anomaly detection task  --**重点看**
- `pretrain.py` - Python script for pretraining MobileUNet  --**不需要 我们用UNI**
- `build_datasets.py` - Python scripts for preprocessing  --**重点看**
- `MobileUNet.pth` - The pre-trained weights for MobileUNet  --**没找到**

In the folder `model`:
- `model/fusion.py` - Multi-modal fusion modules
- `model/meatrd.py` - MEATRD framework for step II and step III
- `model/loss.py` - Loss function of MEATRD
- `model/unet.py` - MobileUNet of MEATRD

