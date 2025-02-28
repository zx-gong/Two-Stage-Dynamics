# Two-Stage-Dynamics
This repository contains the source code of our paper. Majority of the code is borrowed from [LASER](https://github.com/pratyushasharma/laser).

**Disentangling Feature Structure: A Mathematically Provable Two-Stage Training Dynamics in Transformers**

Zixuan Gong, Jiaye Teng, Yong Liu

## Preparation
1. Install the dependencies for our code.
```
pip install -r requirements.txt
```
2. Get the datasets for experiments.
```
python scripts/get_counterfact.py
python scripts/get_hotpot.py
```

## Run a sample code
1. Enter into the directory src/code_counterfact
2. Verifying Two-Stage Learning: Run `python gpt2_counterfact_train_original.py --lnum -1`
3. Verify Spectral Characteristics: Run `python gpt2_counterfact_train_modifyModel.py --lname attn --rate 9`. Modify the parameter within the range of 0 to 9.
