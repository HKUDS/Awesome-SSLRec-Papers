# Awesome-SSLRec-Papers
[![Awesome](https://awesome.re/badge.svg)](https://github.com/HKUDS/Awesome-SSLRec-Papers)
![](https://img.shields.io/github/last-commit/HKUDS/Awesome-SSLRec-Papers?color=green) 
![](https://img.shields.io/badge/PRs-Welcome-red)
![](https://img.shields.io/github/stars/HKUDS/Awesome-SSLRec-Papers?color=yellow)
![](https://img.shields.io/github/forks/HKUDS/Awesome-SSLRec-Papers?color=lightblue)

A collection of papers and resources about self-supervised learning (**SSL**) for recommendation (**Rec**).

Recommender systems personalize suggestions to combat information overload. Deep learning methods like RNNs, GNNs, and Transformers have improved these systems by understanding user behavior better. However, supervised learning struggles with data sparsity. Self-supervised learning (SSL) overcomes this by using inherent data structures for supervision, reducing dependence on labeled data. SSL-based recommender systems accurately predict and recommend, even with sparse data, by leveraging unlabeled data for meaningful representations.

<p align="center">
<img src="fig/taxonomy.png" alt="Framework" />
</p>

## News
🤗 We're actively working on this project, and your interest is greatly appreciated! To keep up with the latest developments, please consider hit the **STAR** and **WATCH** for updates.
* Our survey paper: [A Comprehensive Survey of Self-Supervised Learning for Recommendation]() is now public.

## Overview
This repository serves as a collection of recent advancements in employing self-supervised learning (SSL) across **nine** diverse recommendation scenarios, such as Collaborative Filtering, Sequential Recommendation, and more. We categorize and summarize the approaches based on three primary self-supervised frameworks: *1) Contrastive Learning*, *2) Generative Learning*, and *3) Adversarial Learning*.

- Contrastive Learning <p align="center">
<img src="fig/cl_paradigm.png" alt="Contrastive Learning"/>
</p>

- Generative Learning <p align="center">
<img src="fig/gl_paradigm.png" alt="Contrastive Learning"/>
</p>

- Adversarial Learning <p align="center">
<img src="fig/al_paradigm.png" alt="Contrastive Learning"/>
</p>


We hope this repository proves valuable to your research or practice in the field of self-supervised learning for recommendation systems. If you find it helpful, please consider citing our work:
```bibtex
@article{SSL4RecSys,
  title={A Comprehensive Survey on Self-Supervised Learning for Recommendation},
  author={Ren, Xubin and Wei, Wei and Xia, Lianghao and Huang, Chao},
  journal={arXiv},
  year={2024}
}
```

## Related Resources
* (WSDM'2024) SSLRec: A Self-Supervised Learning Framework for Recommendation [[paper](https://arxiv.org/abs/2308.05697)]
* (TKDE'2023) Self-Supervised Learning for Recommender Systems: A Survey [[paper](https://ieeexplore.ieee.org/abstract/document/10144391)]
* (TOIS'2023) Contrastive Self-supervised Learning in Recommender Systems: A Survey [[paper](https://dl.acm.org/doi/abs/10.1145/3627158)]

## General Collaborative Filtering
### Contrastive Learning
- (arXiv'2021) Contrastive Learning for Recommender System [[paper](https://arxiv.org/abs/2101.01317)]
- (CIKM'2021) SimpleX: A Simple and Strong Baseline for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482297)]
- (SIGIR'2021) Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization [[paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462928)]
- (SIGIR'2021) Self-supervised Graph Learning for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462862)]
- (WSDM'2021) Bipartite Graph Embedding via Mutual Information Maximization [[paper](https://dl.acm.org/doi/abs/10.1145/3437963.3441783)]
- (DASFAA'2021) Diffusion-Based Graph Contrastive Learning for Recommendation with Implicit Feedback [[paper](https://link.springer.com/chapter/10.1007/978-3-031-00126-0_15)]
- (KDD'2022) Towards Representation Alignment and Uniformity in Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539253)]
- (KDD'2022) Self-Supervised Hypergraph Transformer for Recommender Systems [[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539473)]
- (SIGIR'2022) Hypergraph Contrastive Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3532058)]
- (SIGIR'2022) Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531889)]
- (SIGIR'2022) Are Graph Augmentations Necessary?: Simple Graph Contrastive Learning for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531937)]
- (WWW'2022) Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning [[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512104)]
- (ICLR'2023) LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation [[paper](https://arxiv.org/abs/2302.08191)]
- (KDD'2023) Adaptive Graph Contrastive Learning for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599768)]
- (NeurIPS'2023) Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html)]
- (SIGIR'2023) AdaMCL: Adaptive Fusion Multi-View Contrastive Learning for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591632)]
- (SIGIR'2023) Candidate-aware Graph Contrastive Learning for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591647)]
- (SIGIR'2023) Disentangled Contrastive Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591665)]
- (SIGIR'2023) uCTRL: Unbiased Contrastive Representation Learning via Alignment and Uniformity for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3592076)]
- (SIGIR'2023) Generative-Contrastive Graph Learning for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591691)]
- (TKDE'2023) XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation [[paper](https://ieeexplore.ieee.org/abstract/document/10158930)]
- (TOIS'2023) Towards Robust Neural Graph Collaborative Filtering via Structure Denoising and Embedding Perturbation [[paper](https://dl.acm.org/doi/full/10.1145/3568396)]
- (TORS'2023) SelfCF: A Simple Framework for Self-supervised Collaborative Filtering [[paper](https://dl.acm.org/doi/full/10.1145/3591469)]
- (WSDM'2023) Disentangled Negative Sampling for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3539597.3570419)]
- (WSDM'2023) SGCCL: Siamese Graph Contrastive Consensus Learning for Personalized Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3539597.3570422)]
- (WWW'2024) RecDCL: Dual Contrastive Learning for Recommendation [[paper](https://arxiv.org/abs/2401.15635)]

### Generative Learning
- (KDD'2017) Collaborative Variational Autoencoder for Recommender Systems [[paper](https://dl.acm.org/doi/abs/10.1145/3097983.3098077)]
- (WWW'2018) Variational Autoencoders for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3178876.3186150)]
- (NeurIPS'2019) Learning Disentangled Representations for Recommendation [[paper](https://proceedings.neurips.cc/paper/2019/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html)]
- (WSDM'2020) RecVAE: A New Variational Autoencoder for Top-N Recommendations with Implicit Feedback [[paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371831)]
- (WSDM'2021) Bilateral Variational Autoencoder for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3437963.3441759)]
- (WWW'2022) Fast Variational AutoEncoder with Inverted Multi-Index for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512068)]
- (WWW'2022) Mutually-Regularized Dual Collaborative Variational Auto-encoder for Recommendation Systems [[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512110)]
- (WWW'2022) Stochastic-Expert Variational Autoencoder for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512120)]
- (SIGIR'2023) Causal Disentangled Variational Auto-Encoder for Preference Understanding in Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591961)]
- (SIGIR'2023) Diffusion Recommender Model [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591663)]
- (SIGIR'2023) Graph Transformer for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591723)]
- (WWW'2023) Automated Self-Supervised Learning for Recommendation [[paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583336)]

### Adversarial Learning
- (SIGIR'2017) IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models [[paper](https://dl.acm.org/doi/abs/10.1145/3077136.3080786)]
- (CIKM'2018) CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks [[paper](https://dl.acm.org/doi/abs/10.1145/3269206.3271743)]
- (CIKM'2018) An Adversarial Approach to Improve Long-Tail Performance in Neural Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3269206.3269264)]
- (AAAI'2019) Adversarial Binary Collaborative Filtering for Implicit Feedback [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4460)]
- (KDD'2019) Enhancing Collaborative Filtering with Generative Augmentation [[paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330873)]
- (WWW'2019) Rating Augmentation with Generative Adversarial Networks towards Accurate Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3308558.3313413)]
- (CIKM'2020) Exploring Missing Interactions: A Convolutional Generative Adversarial Network for Collaborative Filtering [[paper](https://dl.acm.org/doi/abs/10.1145/3340531.3411917)]

##
