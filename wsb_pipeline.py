#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import fasttext
import fasttext.util
from sqlalchemy import create_engine
from tqdm import tqdm


def lookup_sentence_embedding(text):
    tokens = text.split(" ")
    word_vectors = np.array([ft.get_word_vector(t) for t in tokens])
    return word_vectors.mean(axis=0).tolist()


def create_comment_vectors(chunk):

    cleaned_chunk = pd.DataFrame(
        {
            "date": pd.to_datetime(
                chunk["created_utc"], unit="s", origin="unix"
            ).dt.date,
            "body": chunk["body"],
            "embeddings": chunk["body"].apply(lookup_sentence_embedding),
        }
    )
    return cleaned_chunk


def get_all_embeddings(ticker):
    q = f"""
        SELECT date, embeddings 
        from wsb
        where '{ticker}' LIKE body
        order by date
    """
    db = create_engine(DATABASE_URI)
    df = pd.read_sql_query(q, db)
    return df


DATABASE_URI = "sqlite:///ft_database.db"
if __name__ == "__main__":
    chunksize = 10 ** 4
    db = create_engine(DATABASE_URI)

    ft = fasttext.load_model("../../fastText/cc.en.300.bin")
    fasttext.util.reduce_model(ft, 50)

    wsb = pd.read_json(
        "/home/aaruran/Documents/Git/wsbData.json", lines=True, chunksize=chunksize
    )

    for chunk in tqdm(wsb):
        cleaned_chunk = create_comment_vectors(chunk)
        cleaned_chunk.to_sql("wsb", db, if_exists="append", index=False)
