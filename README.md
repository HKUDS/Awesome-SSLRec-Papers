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