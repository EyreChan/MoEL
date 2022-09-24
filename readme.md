MoEL: Mixture of Empathetic Listeners
====
This repository contains MindSpore implementation of the paper: **MoEL: Mixture of Empathetic Listeners.** Zhaojiang Lin, Andrea Madotto, Jamin Shin, Peng Xu, Pascale Fung *EMNLP 2019* [PDF](https://arxiv.org/pdf/1908.07687.pdf).
This code refers to the [code](https://github.com/HLTCHKUST/MoEL).
Setup
---
Install the required libraries (Python 3.7.5 | CUDA 11.1) and run the command:
`pip install mindspore-cuda11-dev -i https://pypi.tuna.tsinghua.edu.cn/simple`
Download [Pretrained GloVe Embeddings](http://nlp.stanford.edu/data/glove.6B.zip)((glove.6B.300d.txt)) and save it in /vectors.
Download dataset from [https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue](https://github.com/HLTCHKUST/MoEL/tree/master/empathetic-dialogue) and save it in /empathetic-dialogue.
Training & Testing
---
`python3 main.py --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --topk 5 --cuda --pretrain_emb --basic_learner --schedule 10000 --save_path save/moel/`
