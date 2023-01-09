# Market Aware Models for Efficient Cross Market Recommendation (ECIR 2023)

Code for the paper "Market Aware Models for Efficient Cross Market Recommendation" (ECIR 2023)


## About the paper
**Authors**: [Samarth Bhargav](https://samarthbhargav.github.io/), [Mohammad Aliannejadi](https://aliannejadi.com/), 
[Evangelos Kanoulas](https://staff.fnwi.uva.nl/e.kanoulas/) 

**Abstract**

```
We consider the Cross-Market Recommendation (CMR) task, which involves recommendation in a low 
resource target market using data from a richer, auxiliary source market. Prior work in CMR 
utilized meta-learning to improve recommendation performance in target markets; meta-learning 
however can be complex and time-consuming. In this paper we propose Market Aware (MA) models, 
which directly models the market via market embeddings instead of meta-learning across
markets. These embeddings transform item representations into market-specific representations.
Our experiments highlight the effectiveness and efficiency of MA models both in a pairwise setting
with a single target-source market, as well as a global model trained on all markets in unison.
In the former pairwise setting, MA models on average outperform market-unaware models in 85% of 
cases on nDCG@10, while being time-efficient - compared to meta-learning models, MA models require 
only 15% of the training time. In the global setting, MA models outperform market-unaware models 
consistently for some markets, while outperforming meta-learning-based methods for all but one
market. We conclude that MA models are an efficient and effective alternative to
meta-learning, especially in the global setting.
```

**Citation**
```
todo!
```

**Contact**

We're happy to help with reproducability and other questions. 
Reach out via email, which can be found at our respective websites: [Samarth Bhargav](https://samarthbhargav.github.io/) 
(corresponding author), [Mohammad Aliannejadi](https://aliannejadi.com/), 
[Evangelos Kanoulas](https://staff.fnwi.uva.nl/e.kanoulas/)

## Reproducing results

### Environment setup

1. Install `python 3.7.10`. We recommend using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
2. Install requirements via `pip install -r requirements.txt`

### Reproducing experiments
1. Run the commands experiments using the instructions in [RUN.md](RUN.md)
2. Create a directory `raw_results` at the repo root, and move `forec_eval_single`, `forec_eval_all`, 
`forec_eval_all_market_aware`, and `forec_single_model` into `raw_results`.
3. Run [the results nb](results.ipynb) 



## Code Acknowledgements
- Most of the code builds on the [original XMRec code](https://github.com/hamedrab/FOREC). Thanks Hamed!