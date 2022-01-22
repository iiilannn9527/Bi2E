# Bi2E: Bidirectional Knowledge Graph Embeddings Based on  Subject-Object Feature Spaces.

Code for paper [Bi2E: Bidirectional Knowledge Graph Embeddings Based on Subject-Object Feature Spaces](https://openreview.net/pdf?id=weNI9o5Sgf).

## Setup
These experiments are based on [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
Adjust the model's hyperparameters by setting [arguments.py](https://github.com/iiilannn9527/Bi2E/blob/master/arguments.py) and run in [run.py](https://github.com/iiilannn9527/Bi2E/blob/master/run.py).

To run the code, you need the following dependencies:
- [Pytorch 1.6.0](https://pytorch.org/)
## Results
The results of **Bi2E** on **WN18RR**, **YAGO3-10** and **FB15k-237** are as follows.
| | MR| MRR| Hits@1| Hits@3| Hits@10|
|:------:|:------:|:------:|:------:|:--------:|:--------:|
| WN18RR | 2798 | 0.480 | 0.432 | 0.498 | 0.574 | 
| YAGO3-10 | 1496 | 0.550 | 0.468 | 0.603 | 0.697 |
|FB15k-237| 169| 0.346| 0.249| 0.384|0.544|

Hyper-parameters to reproduce the reuslts are set in [arguments.py](https://github.com/iiilannn9527/Bi2E/blob/master/arguments.py)


<!-- WN18RR
```
{"learn rate": 0.00025, "max_steps": 120000, "warm_up_steps": 30000, "valid_steps": 120000, "log_steps": 10000,
 "damping_steps": 30000, "damping_rate": 2.0,
 "negative_adversarial_sampling": True, "negative sample size": 256,
 "train batch size": 512, "test batch size": 32,
 "hidden dim": 500, "gamma": 2.5,
 "adversarial_temperature": 1.5}
```

FB15k-237
```
{"learn rate": 0.00025, "max_steps": 120000, "warm_up_steps": 30000, "valid_steps": 120000, "log_steps": 10000,
 "damping_steps": 30000, "damping_rate": 2.0,
 "negative_adversarial_sampling": True, "negative sample size": 256,
 "train batch size": 1024, "test batch size": 16,
 "hidden dim": 1500, "gamma": 7.5,
 "adversarial_temperature": 1.25}
``` -->
