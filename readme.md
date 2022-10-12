MoEL: Mixture of Empathetic Listeners
====

This repository contains the MindSpore implementation of the paper: 

**MoEL: Mixture of Empathetic Listeners.**  Zhaojiang Lin, Andrea Madotto, Jamin Shin, Peng Xu, Pascale Fung. *EMNLP 2019*. [PDF](https://arxiv.org/pdf/1908.07687.pdf).

And this code refers to the [PyTorch implementation](https://github.com/HLTCHKUST/MoEL) from paper authors.

| 测试 | test |

Setup
---

1. Install the required libraries (Python 3.7.5 | CUDA 11.1)
2. Run the command: `pip install mindspore-cuda11-dev -i https://pypi.tuna.tsinghua.edu.cn/simple`

3. Download [Pretrained GloVe Embeddings](http://nlp.stanford.edu/data/glove.6B.zip) (glove.6B.300d.txt) and save it in `/vectors`.
4. Download dataset from [https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue](https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue) and save it in `/empathetic-dialogue`.

Training & Testing
---

`python3 main.py --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --topk 5 --cuda --pretrain_emb --basic_learner --schedule 10000`

Predicted Value
---

BLEU: `2.5`
