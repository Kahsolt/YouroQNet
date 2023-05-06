# VQNet-text-sentiment-clf

    A toy text sentiment classifier using pyVQNet and pyQPanda

----

This repo contains code for the final problem of the OriginQ's [2nd CCF "Pilot Cup" contest](https://contest.originqc.com.cn/contest/4/contest:introduction) (Professional Group - Quantum Machine Learning Track).

And, code repo for the qualifying stage is here: [第二届“司南杯”初赛](https://github.com/Kahsolt/CCF-2nd-Pilot-Cup-first-stage)


### Quickstart

⚪ install

  - `conda create -n q python==3.8` (pyvqnet requires Python 3.8)
  - `conda activate q`
  - `pip install -r requirements.txt`

⚪ for evaluation only

  - `run_quantum.cmd`

⚪ for full development

  - `pushd repo & init_repos.cmd & popd` for extra dependencies
    - fasttext==0.9.2 requires numpy<1.24 (things might changed)
  - `python mk_stats.py` for dataset stats & plots
  - `run_baseline.cmd` for classic models
  - `run_quantum.cmd` for quantum models


#### Dataset

A subset from [simplifyweibo_4_moods](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb): `1600` samples for train, `400` samples for test. Class label names: `0 - joy`, `1 - angry`, `2 - hate`, `3 - sad`, however is not very semantically corresponding in the datasets :(

⚠ File naming rule: `train.csv` is train set, `test.csv` is valid set, and the generated `valid.csv` might be the real test set for this contest. **We use csv filename to refer to each split in the code**


### Todo List

- [x] data exploration
  - [x] guess the target test set (`valid.txt`)
  - [x] vocab & freq stats
  - [x] pca & cluster
- [ ] data filtering
  - [ ] stop words
  - [ ] too short / long sententce
- [x] feature extraction
  - [x] tf-idf (syntaxical)
  - [x] fasttext embedding (sematical)
- [x] baseline models
  - [x] sklearn
  - [ ] vqnet-classical
- [x] quantum models
  - [ ] quantum embedding
  - [ ] model route on different length
  - [ ] multi to binary clf
  - [ ] contrastive learning
  - [ ] learn the difference


### references

- NLP clf survey: [https://zhuanlan.zhihu.com/p/349086747](https://zhuanlan.zhihu.com/p/349086747)
- fastText: [https://github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText)
- QSANN: [https://arxiv.org/abs/2205.05625](https://arxiv.org/abs/2205.05625)
- QNLP-DisCoCat: [https://arxiv.org/pdf/2102.12846.pdf](https://arxiv.org/pdf/2102.12846.pdf)
  - repo: [https://github.com/CQCL/qnlp_lorenz_etal_2021_resources](https://github.com/CQCL/qnlp_lorenz_etal_2021_resources)

----

by Armit
2023/05/03 
