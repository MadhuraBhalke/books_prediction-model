import streamlit as st
import pandas as pd
import pickle 
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
st.title('Books Recommender System')


books_dict = pickle.load(open('books_model.pkl' , 'rb'))

books = pd.DataFrame(books_dict)

options = st.selectbox('What would you like to read today ?', books['title'].values )
df = pd.read_csv(r'C:\Users\jayes\Downloads\archive\books.csv',error_bad_lines = False)


df2 = df.copy()

df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])

features = pd.concat([rating_df, language_df, df2['average_rating'], df2['ratings_count']], axis=1)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)

def BookRecommender(book_name):
    book_list_name = []
    book_id = books[books['title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].title)
    return book_list_name
books_algo = pickle.load(open('books_model_algo.pkl' , 'rb'))



if st.button('Recommend'):
    recommendations = BookRecommender(options)
    for i in recommendations:
        st.write(i)