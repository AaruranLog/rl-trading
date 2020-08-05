import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import fasttext
import fasttext.util

chunksize=10**1
DATABASE_URI = 'sqlite:///ft_database.db'

def create_comment_vectors(chunk):
    chunk = chunk[['created_utc', 'body']]
    chunk['date'] = pd.to_datetime(chunk['created_utc'], unit='s',
                                                  origin='unix')
    chunk.drop('created_utc', axis=1, inplace=True)
    chunk['date'] = chunk['date'].dt.date
    chunk['embeddings'] = chunk['body'].apply(lookup_sentence_embedding)
    return chunk

def lookup_sentence_embedding(text):
    tokens = text.split(' ')
    word_vectors = np.array([ft.get_word_vector(t) for t in tokens])
    print(word_vectors.shape)
    return word_vectors.mean(axis=0)

if __name__ == "__main__":
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 50)
    
	db = create_engine(DATABASE_URI)

    wsb = pd.read_json('/home/aaruran/Documents/Git/wsbData.json',
                       lines=True, chunksize=chunksize)
    create_comment_vectors(next(wsb))
#     for chunk in tqdm(wsb):
#         cleaned_chunk = create_comment_vectors(chunk)
#         cleaned_chunk.to_sql('wsb', db, if_exists='append', index=False)