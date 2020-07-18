import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from wsb_pipeline import DATABASE_URI, get_all_embeddings


def test_basic_query():
    db = create_engine(DATABASE_URI, pool_pre_ping=True)

    conn = sqlite3.connect("ft_database.db")
    c = conn.cursor()

    queries = pd.read_sql_query(
        "select body from wsb where date = '2016-01-01' limit 1", conn
    )

def test_wsb_db_aapl():
    _ = get_all_embeddings('AAPL')

def test_wsb_db_tsla():
    _ = get_all_embeddings('TSLA')
    

