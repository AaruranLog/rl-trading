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


# working_dir = pathlib.Path.cwd()  # this should always end in /rl-trading/trading
db_file = pathlib.Path('/home/aaruran/Documents/Git/rl-trading/ft_database.db')
DATABASE_URI = "sqlite:///" + db_file.as_posix()

if __name__ == "__main__":
    if not db_file.is_file():
        print('File ft_database.db not found. Processing...')
        db = create_engine(DATABASE_URI, echo=True)
        print('Loading fastText model ...', end='')
        ft_file = pathlib.Path('/home/aaruran/Documents/Git/rl-trading/cc.en.300.bin')
        ft = fasttext.load_model(ft_file.as_posix())
        fasttext.util.reduce_model(ft, 50)
        print('loaded.')
        print('Preparing Reddit comments ...', end='')
        wsb_file = pathlib.Path("/home/aaruran/Documents/Git/rl-trading/wsbData.json")
        chunksize = 10 ** 4

        wsb = pd.read_json(wsb_file.as_posix(), lines=True, chunksize=chunksize)
        print('prepared.')
        print('Writing to file...')

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
            cleaned_chunk["date"] = pd.to_datetime(cleaned_chunk["date"])
            cleaned_chunk[["body", "embeddings"]] = cleaned_chunk[
                ["body", "embeddings"]
            ].astype("str")
            return cleaned_chunk
        for chunk in tqdm(wsb):
            cleaned_chunk = create_comment_vectors(chunk)
            cleaned_chunk.to_sql(
                "wsb",
                db,
                if_exists="append",
                index=False,
                dtype={"date": DateTime, "body": String(600), "embeddings": String(1200)},
            )
        print('done.')
    else:
        print('Database file already exists.')
