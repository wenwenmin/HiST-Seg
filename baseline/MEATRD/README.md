
## Usage
Please download the processed dataset and store it in the ```/data/``` directory, then run:
```
python main.py --data_path ./data/
```

## Code organization

In the folder `model`:
- `model/fusion.py` - Multi-modal fusion modules
- `model/meatrd.py` - MEATRD framework for step II and step III
- `model/loss.py` - Loss function of MEATRD
- `model/unet.py` - MobileUNet of MEATRD

