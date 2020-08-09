import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from trading.data.wsb_pipeline import *
from pandas.testing import assert_frame_equal


def test_basic_query():
    db = create_engine(DATABASE_URI, pool_pre_ping=True)
    queries = pd.read_sql_query(
        "select body from wsb where date = '2016-01-01' limit 1", db
    )


def test_wsb_db_aapl():
    _ = get_all_embeddings("AAPL")

def test_wsb_db_tsla():
    _ = get_all_embeddings("TSLA")

def test_nontrivial_text():
#     DATABASE_URI = "sqlite:///ft_database.db"
    q = f"""
            SELECT date, embeddings 
            from wsb
            where body LIKE '%amzn%'
            order by date
    """
    db = create_engine(DATABASE_URI)
    df = pd.read_sql_query(q, db)

    df2 = get_all_embeddings('AMZN')
    assert_frame_equal(df, df2)