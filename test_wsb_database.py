import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from wsb_pipeline import DATABASE_URI

def test_basic_query():
	# print(pd.read_csv(file, nrows=5))
	csv_database = create_engine(DATABASE_URI, pool_pre_ping=True)

	conn = sqlite3.connect('csv_database.db')
	c = conn.cursor()

	queries = pd.read_sql_query('select body from wsb where created_utc = 2016-01-01 limit 1',conn)
	# pd.read_sql_query('select * from table limit 10', csv_database)

# def test_wsb_db_aapl():
# 	_ = get_embeddings('2016-01-01', 'AAPL')
# 	_ = get_all_embeddings('AAPL')

# def test_wsb_db_tsla():
# 	_ = get_embeddings('2016-01-01', 'TSLA')
# 	_ = get_all_embeddings('TSLA')
