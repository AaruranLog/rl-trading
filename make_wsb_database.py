import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

file = '../../cleaned_wsb.csv'
DATABASE_URI = 'sqlite:///csv_database.db'
# print(pd.read_csv(file, nrows=5))

if __name__ == "__main__":
	csv_database = create_engine(DATABASE_URI)
	chunksize = 10000
	data = pd.read_csv(file, chunksize=chunksize, iterator=True,
		parse_dates=['created_utc'], usecols=['body', 'created_utc', 'embeddings'])

	for df in tqdm(data):
		df.to_sql('wsb', csv_database, if_exists='append', index=False)
