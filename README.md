# VQNet-text-sentiment-clf

    A toy quantum text sentiment classifier using pyVQNet and pyQPanda

----

This repo contains code for the final problem of the OriginQ's [2nd CCF "Pilot Cup" contest](https://contest.originqc.com.cn/contest/4/contest:introduction) (Professional Group - Quantum Machine Learning Track).

And, code repo for the qualifying stage is here: [第二届“司南杯”初赛](https://github.com/Kahsolt/CCF-2nd-Pilot-Cup-first-stage)


### Quickstart

⚪ install

- `conda create -n q python==3.8` (pyvqnet requires Python 3.8)
- `conda activate q`
- `pip install -r requirements.txt`

⚪ for contest problem (👈 **Follow this to reproduce our contest results!!**)

- `python answer.py` for preprocess & train
- `python check.py` for evaluate
- `python vis_tokenizer.py` for adaptive k-gram tokeinzer interactive demo
- `python vis_youroqnet.py` for YouroQNet interactive demo

⚪ for full development

- `pip install -r requirements_dev.txt` for extra dependencies
- `pushd repo & init_repos.cmd & popd` for extra git repos
  - fasttext==0.9.2 requires numpy<1.24 (things might changed)
- `start_shell.cmd` to enter deveolp run command env
  - `start_shell.cmd py` to get a ipy console that quick refering to `pyvqnet`'s fucking undocumented-documentation with `help()`
- `mk_preprocess.cmd` for making clean datasets, stats, plots & vocabs etc... (~7 minutes)
- `python vis_project.py` to see 3d data projection (you will understand what the fuck this dataset is 👿)
- `run_baseline.cmd` to run classic models
- `run_quantum.cmd` to run quantum models


#### Dataset

A subset from [simplifyweibo_4_moods](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb): `1600` samples for train, `400` samples for test. Class label names: `0 - joy`, `1 - angry`, `2 - hate`, `3 - sad`, however is not very semantically corresponding in the datasets :(

⚠ File naming rule: `train.csv` is train set, `test.csv` is valid set, and the generated `valid.csv` might be the real test set for this contest. **We use csv filename to refer to each split in the code**


### Todo List

- [x] data exploration
  - [x] guess the target test set (`valid.txt`)
  - [x] vocab & freq stats
  - [x] pca & cluster
  - [ ] data relabel (?)
- [ ] data filtering
  - [x] punctuation sanitize
  - [ ] stop words removal
  - [x] too short / long sententce
- [x] feature extraction
  - [x] tf-idf (syntaxical)
  - [x] fasttext embedding (sematical)
  - [x] adaptive tokenizer
- [x] baseline models
  - [x] sklearn
  - [x] vqnet-classical
- [ ] quantum models
  - [ ] quantum embedding
  - [ ] model route on different length
  - [ ] multi to binary clf
  - [ ] contrastive learning
  - [ ] learn the difference


### Core idea & Contributions

- adaptive k-gram tokenizer (see [mk_vocab.py](mk_vocab.py), interactivate demo [vis_tokenizer.py](vis_tokenizer.py))
- YouroQNet for text clf (see [run_quantum.py](run_quantum.py), interactivate demo [vis_quantum.py](vis_quantum.py))


### Project layout

```python
# meterials
ref/                # thesis for dev
repo/               # git repo for dev
data/               # dataset
log/                # outputs

# contest related
answer.py           # run script for preprocessing & training
check.py            # run script for evalution

# preprocessors
mk_*.py
mk_preprocess.cmd   # run script for mk_*.py

# models
run_baseline_*.py   # classical experiments
run_baseline.cmd    # run script for run_baseline_*.py
run_quantum.py      # quantum experiments
run_quantum.cmd     # run script for run_quantum.py
run_quantum_toy.py  # toy QNN for debug and verify

# misc
vis_*.py            # intercative demos or debug scaffolds
utils.py            # common utils
start_shell.cmd     # develop env entry

# doc & lic
README.md
TECH.md             # techincal & theoretical stuff
requirements_*.txt
LICESE
```

ℹ For the contest, only these files are submitted: `answer.py`, `mk_vocab.py`, `run_quantum.py`, `utils.py`, `README.md`; it should be enough to run all quantum parts 😀


### references

- text-clf survey:
  - [https://zhuanlan.zhihu.com/p/349086747](https://zhuanlan.zhihu.com/p/349086747)
  - [https://zhuanlan.zhihu.com/p/161068416](https://zhuanlan.zhihu.com/p/161068416)
  - [https://www.cnblogs.com/sandwichnlp/p/11698996.html](https://www.cnblogs.com/sandwichnlp/p/11698996.html)
  - [https://mp.weixin.qq.com/s?__biz=MzI1MjQ2OTQ3Ng==&mid=2247485438&idx=1&sn=00dfcb8c344c3a622a88d9360c866c2e](https://mp.weixin.qq.com/s?__biz=MzI1MjQ2OTQ3Ng==&mid=2247485438&idx=1&sn=00dfcb8c344c3a622a88d9360c866c2e)
- fastText: 
  - Enriching Word Vectors with Subword Information: [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
  - Bag of Tricks for Efficient Text Classification: [https://arxiv.org/abs/1607.01759](https://arxiv.org/abs/1607.01759)
- QNLP-DisCoCat: [https://arxiv.org/pdf/2102.12846.pdf](https://arxiv.org/pdf/2102.12846.pdf)
- QSANN: [https://arxiv.org/abs/2205.05625](https://arxiv.org/abs/2205.05625)

=> find thesis of related work in [ref/init_thesis.cmd](ref/init_thesis.cmd)  
=> find implementations of related work in [repo/init_repos.cmd](repo/init_repos.cmd)  

----

by Armit
2023/05/03 
