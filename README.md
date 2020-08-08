# rl-trading
This repository contains the source code for my project report, for the course CS 885 (Reinforcement Learning), taught by Professor Pascal Poupart.

To install and run, use the file `environment.lock.yaml` and `conda`.

The notebooks contain (very messy) code used to experiment, and develop, the source code in the `trading/` folder.
In particular, `notebooks/experiments.ipynb` generated most of the illustrations in the `plots/` folder.

## Getting the Reddit Comments

This part of data pipeline is intentionally not automatic.

To acquire the Reddit comments dataset from [Christopher Lambert's Kaggle release](https://www.kaggle.com/theriley106/wallstreetbetscomments):

```
> wget "https://www.kaggle.com/theriley106/wallstreetbetscomments?select=wsbData.json"
```
You will need to download the Common Crawl word vectors. [See here for how that can be done](https://fasttext.cc/docs/en/crawl-vectors.html).
Store the unzipped file in the top-level of this cloned repository.
Run the script `trading/wsb_pipeline.py` to create a database of comment vectors. These embeddings are used to train `ModelBasedAgent` objects,
as they are processed through `TradingWithRedditEnv`.
