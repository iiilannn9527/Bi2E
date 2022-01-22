# Bi2E: Bidirectional Knowledge Graph Embeddings Based on  Subject-Object Feature Spaces.

Code for paper [Bi2E: Bidirectional Knowledge Graph Embeddings Based on Subject-Object Feature Spaces](https://openreview.net/pdf?id=weNI9o5Sgf).

## Setup
These experiments are based on [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

Adjust the model's hyperparameters by setting [arguments.py](https://github.com/iiilannn9527/Bi2E/blob/master/arguments.py) and run in [run.py](https://github.com/iiilannn9527/Bi2E/blob/master/run.py).

To run the code, you need the following dependency:
- [Pytorch 1.6.0](https://pytorch.org/)
## Results
The results of **Bi2E** on **WN18RR**, **YAGO3-10** and **FB15k-237** are as follows.
| | MR| MRR| Hits@1| Hits@3| Hits@10|
|:------:|:------:|:------:|:------:|:--------:|:--------:|
| WN18RR | 2798 | 0.480 | 0.432 | 0.498 | 0.574 | 
| YAGO3-10 | 1496 | 0.550 | 0.468 | 0.603 | 0.697 |
|FB15k-237| 169| 0.346| 0.249| 0.384|0.544|

## Implementation
Hyper-parameters to reproduce the reuslts are set in [arguments.py](https://github.com/iiilannn9527/Bi2E/blob/master/arguments.py)

Bi2E is implemented by PyTorch and runs on a NVIDIA RTX-3090. Bi2E needs different max_steps to converge on different dataset:
| dataset| WN18RR| YAGO3-10| FB15k-237|
|:------:|:------:|:------:|:------:|
| MAX_STEPS | 120,000 | 150,000 | 120,000 | 
| Time | 2h | 4.5h | 5h |
