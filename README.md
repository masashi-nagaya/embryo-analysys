# embryo-analysys

# Environment

python : 3.6.9

PyTorch : 1.4.0

PyTorch vision : 0.5.0
 
# Installation

Installing PyText and making virtualenv

how to install Pytext:

https://github.com/facebookresearch/pytext

after making virtualenv, install requirement.txt

```bash
pip install -r requirements.txt
```
 
# Execution

## frame-wise
### 1. make csv
```bash
python make_csv.py --time 25 --data_dir "/home/masashi_nagaya/M2/dataset_9_08/all/"
```

### 2. train

#### pn classification
```bash
python abn_pn.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50 
```

#### nnpu classification
```bash
python abn_pu.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50
```

#### auc-pr optimization
```bash
python abn_auc-pr.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50
```

#### auc-pr-nnpu optimization
```bash
python abn_auc-pr-pu.py --resume ./checkpoint/imagenet/resnet50/checkpoint.pth.tar --val_num 1 --manualSeed 1 –a resnet50
```

when mutual information maximization, change codes as follows: 
 - PN：abn_pn.py→abn_pn_iic.py
 - nnPU：abn_pu.py→abn_pu_iic.py
 - AUC-PR：abn_auc-pr.py→abn_auc_pr_iic.py
 - AUC-PR-nnPU：auc-pr-pu.py→ auc-pr-pu_iic.py

### 3. test
#### all-frame evaluation
```bash
python test.py --resume ./checkpoint/pn_mse1.0_seed --val_num 1　--seed_number 5 --mode pn --eval all
```
#### final-frame evaluation
```bash
python test.py --resume ./checkpoint/pn_mse1.0_seed --val_num 1　--seed_number 5 --mode pn --eval last
```
#### weighted average evaluation
```bash
python test_weighted.py --resume ./checkpoint/pn_mse1.0_seed --val_num 1 --seed_number 5 --mode pn
```
#### attention map output
```bash
python attention.py --resume ./checkpoint/pn_mse1.0_seed1.pth.tar --val_num 1 --test_mode pn
```

## Recurrent

### 1. make csv
```bash
python make_csv.py
```
### 2. feature extraction
```bash
python extract_aug.py --val_num 1 --resume ./checkpoint/pn1_mse1.0_seed1.pth.tar 
```
### 3. train
#### final-state feature
```bash
python gru_final.py --val_num 1 --manualSeed 1
```
#### temporal-pool feature
```bash
python gru_temporal.py --val_num 1 --manualSeed 1
```
### 4. test
```bash
python gru_eval.py --val_num 1 --resume ./checkpoint/gru1_seed --arch final 
```

## Citation
Please cite this [paper](https://ieeexplore.ieee.org/document/9606688) in your publications if this dataset helps your research.

```
@article{
  title={Embryo grading from unreliable labels by positive-unlabeled classification with ranking},
  author={Masashi Nagaya and Norimichi Ukita},
  journal={{IEEE} Transactions on Medical Imaging},
  volume    = {41},
  number    = {2},
  pages     = {320--331},
  year={2022}
}
```


