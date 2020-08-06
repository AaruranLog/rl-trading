#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import fasttext
import fasttext.util
from sqlalchemy import create_engine
from sqlalchemy.types import DateTime, String
from tqdm import tqdm
import pathlib

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
            "embeddings": chunk["body"].apply(lookup_sentence_embglanedding),
        }
    )
    cleaned_chunk['date'] = pd.to_datetime(cleaned_chunk['date'])
    cleaned_chunk[['body', 'embeddings']] = cleaned_chunk[['body', 'embeddings']].astype('str')
    return cleaned_chunk


def get_all_embeddings(ticker):
    q = f"""
        SELECT date, embeddings 
        from wsb
        where body LIKE '%{ticker}%'
        order by date
    """
    db = create_engine(DATABASE_URI)
    df = pd.read_sql_query(q, db)
    return df


working_dir = pathlib.Path.cwd() # this should always end in /rl-trading/trading
db_file = (working_dir.parent / 'ft_database.db').as_posix()
# DATABASE_URI = "sqlite:///../ft_database.db"
DATABASE_URI = 'sqlite:///' + db_file

if __name__ == "__main__":
    chunksize = 10 ** 3
    db = create_engine(DATABASE_URI, echo=True)
    ft_file = working_dir.parent.parent / 'fastText' / 'cc.en.300.bin'
    ft = fasttext.load_model(ft_file.as_posix())
    fasttext.util.reduce_model(ft, 50)

    wsb_file = working_dir.resolve().parent.parent / 'wsbData.json'
    wsb = pd.read_json(wsb_file.as_posix(), lines=True, chunksize=chunksize)

    for chunk in tqdm(wsb):
        cleaned_chunk = create_comment_vectors(chunk)
        cleaned_chunk.to_sql("wsb", db, if_exists="append", index=False,
                            dtype={
                                "date" : DateTime,
                                "body" : String(600),
                                "embeddings" : String(1200)
                            })
