import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import ast
import re
from make_wsb_database import DATABASE_URI

DB_CONN = create_engine(DATABASE_URI, pool_pre_ping=True)

def get_all_embeddings(ticker):
    query = f"""
		SELECT created_utc as date, embeddings FROM wsb
		where body LIKE '%{ticker}%'
		order by created_utc
	"""
    df = pd.read_sql_query(query, DB_CONN)
    df["embeddings"] = df["embeddings"].apply(lambda s: s.replace("\n", "").rstrip()).tolist()

    # vectors = [
    # 		np.array(ast.literal_eval(re.sub("\s+", ",",s))) for s in values
    # ]
    
    return df

def get_embeddings(date, ticker):
    query = f"""
			SELECT embeddings FROM wsb
			where body LIKE '%{ticker}%'
			and created_utc = '{date}'
		"""
    df = pd.read_sql_query(query, DB_CONN)
    # values = df['embeddings'].apply(lambda s : s.replace('\n', '').rstrip()).tolist()
    values = df["embeddings"].tolist()
    vectors = np.array([(ast.literal_eval(re.sub("\s+", ",", s))) for s in values])
    return vectors


# get_embeddings('2016-01-01', 'AAPL')
# get_embeddings('2016-01-01', 'TSLA')

# date = "2016-01-01"
# ticker = "AAPL"
# query = f'''
# 		SELECT embeddings FROM wsb
# 		where body LIKE '%{ticker}%'
# 		and created_utc = '{date}'
# 	'''
# df = pd.read_sql_query(query, DB_CONN)
# # values = df['embeddings'].apply(lambda s : s.replace('\n', '').rstrip()).tolist()
# values = df['embeddings'].tolist()

# vectors = np.array([
# 		(ast.literal_eval(re.sub("\s+", ",",s))) for s in values
# ])

# print(get_embeddings(date, ticker))
