# README

## Description
This project implements several recommendation system algorithms and an evaluation framework for the patient-narritive dataset.
Specifically, Random, UserCF, ItemCF and Pixie[1] are already implemented. The metrics are Mean Reciprocal Rank (MAP), Mean Average Precision (MAP) and Normalized
Discount Cummulative Gain (NDCG). 

## Usage
```python
python main.py
```
The main.py will run several experiments on the four policies and evaluate the average performance.
The result is
```bash
$ python main.py
100%|████████████████████████████████████████████████████████████████| 50/50 [00:48<00:00,  1.04s/it]

Method	MRR
Random	0.091
UserCF	0.166
ItemCF	0.075
Pixie	0.171

Method	MAP
Random	0.011
UserCF	0.012
ItemCF	0.009
Pixie	0.009

Method	NDCG
Random	1.175
UserCF	1.378
ItemCF	1.131
Pixie	1.359
```

[1] Pixie: A system for Recommending 1+ Billion Items to 175+ Million Pinterest Users in Real-Time