#rl-trading

## Getting the Reddit Comments

To acquire the Reddit comments dataset from [Christopher Lambert's Kaggle release](https://www.kaggle.com/theriley106/wallstreetbetscomments):

```
	wget "https://www.kaggle.com/theriley106/wallstreetbetscomments?select=wsbData.json"
```
To generate text embeddings, you must install the fastText library (into a virtualenv, conda environment, etc.)
[See details on installing the fastText python package here](https://github.com/facebookresearch/fastText#building-fasttext-for-python).

You will also need to download the Common Crawl word vectors. [See here for how that can be done](https://fasttext.cc/docs/en/crawl-vectors.html).
Store the unzipped file in your fastText repository.