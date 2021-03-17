# embryo-analysys

 
# Requirement
 
"hoge"を動かすのに必要なライブラリなどを列挙する
 
* huga 3.5.2
* hogehuga 1.0.2
 
# Installation
 
```bash
pip install huga_package
```
 
# Execution

# frame-wise
## 1. make csv
```bash
python make_csv.py --time 25 --data_dir "/home/masashi_nagaya/M2/dataset_9_08/all/"
```

## 2. train

### pn classification
```bash
python abn_pn.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50 
```

### nnpu classification
```bash
python abn_pu.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50
```

### auc-pr optimization
```bash
python abn_auc-pr.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50
```

### auc-pr-nnpu optimization
```bash
python abn_auc-pr-pu.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50
```

when mutual information maximization, change codes as follows: 
 - PN：abn_pn.py→abn_pn_iic.py
 - nnPU：abn_pu.py→abn_pu_iic.py
 - AUC-PR：abn_auc-pr.py→abn_auc_pr_iic.py
 - AUC-PR-nnPU：auc-pr-pu.py→ auc-pr-pu_iic.py

