# Import libraries
# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
pd.pandas.set_option('display.max_columns', None)

# Load datasets
movies_data = pd.read_csv('./data/movies_metadata.csv')
link_small = pd.read_csv('./data/links_small.csv')
credits = pd.read_csv('./data/credits.csv')
keyword = pd.read_csv('./data/keywords.csv')

# Remove rows with specific index labels
movies_data = movies_data.drop([19730, 29503, 35587])
movies_data['id'] = movies_data['id'].astype('int')
link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# Filter and process data
smd = movies_data[movies_data['id'].isin(link_small)]
smd['tagline'] = smd['tagline'].fillna('')
smd['overview'] = smd['overview'].fillna('')
keyword['id'] = keyword['id'].astype('int')
credits['id'] = credits['id'].astype('int')

# Merge datasets
movies_data_merged = movies_data.merge(keyword, on='id')
movies_data_merged = movies_data_merged.merge(credits, on='id')
smd2 = movies_data_merged[movies_data_merged['id'].isin(link_small)]

# Data processing
smd2['cast'] = smd2['cast'].apply(literal_eval)
smd2['crew'] = smd2['crew'].apply(literal_eval)
smd2['keywords'] = smd2['keywords'].apply(literal_eval)
smd2['cast_size'] = smd2['cast'].apply(lambda x: len(x))
smd2['crew_size'] = smd2['crew'].apply(lambda x: len(x))

def get_director(names):
    for i in names:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd2['directors'] = smd2['crew'].apply(get_director)
smd2['directors'] = smd2['directors'].astype('str')
smd2['directors'] = smd2['directors'].apply(lambda x: [x, x, x])

smd2['cast'] = smd2['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd2['cast'] = smd2['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd2['keywords'] = smd2['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd2['overview'] = smd2['overview'].apply(lambda x: [x] if isinstance(x, str) else x)

s = smd2.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

def keywords(x):
    m = []
    for i in x:
        if i in s:
            m.append(i)
    return m

stemmer = SnowballStemmer('english')
smd2['keywords'] = smd2['keywords'].apply(keywords)

from itertools import chain
def process(x):
    it = [["".join(y) for y in i.split(" ")] for i in x]
    return list(chain.from_iterable(it))

smd2["keywords"] = smd2['keywords'].apply(process)

test1 = smd2.copy()
# Convert overview to list to avoid concatenation errors
test1['overview'] = test1['overview'].apply(lambda x: [x] if isinstance(x, str) else [])
test1["soup"] = smd2['keywords'] + smd2['cast'] + smd2['directors'] + smd2['overview']
# Apply join only if 'x' is iterable; otherwise, convert to empty string or handle non-iterables
test1['soup'] = test1['soup'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
test1['soup'] = test1['soup'].apply(lambda x: remove_stopwords(x))

test1['soup'] = test1['soup'].apply(lambda x: x.split(" ")).apply(lambda x: [stemmer.stem(i) for i in x])

d = {item: "NA" for item in ['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id', 
                             'original_language', 'original_title', 'overview', 'popularity', 'poster_path', 
                             'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 
                             'spoken_languages', 'status', 'tagline', 'title', 'video', 'vote_average', 'vote_count', 
                             'keywords', 'cast', 'crew', 'cast_size', 'crew_size', 'description', 'directors', 'soup']}

def getPredictionsV2(soup, smd, num):
    smd = smd[smd.title != "myRequest"]
    stopword_removed_soup = remove_stopwords(soup)
    soup_list = stopword_removed_soup.split(" ")
    soup_list_stemmed = [stemmer.stem(i) for i in soup_list]
    stemmed_soup = " ".join(soup_list_stemmed)

    d["soup"] = stemmed_soup
    d["title"] = "myRequest"
    smd = pd.concat([smd, pd.DataFrame([d])], ignore_index=True)

    smd['soup'] = smd['soup'].fillna('')

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfid_mat = tf.fit_transform(smd['soup'])
    cos_sim = linear_kernel(tfid_mat, tfid_mat)

    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])

    idx = indices["myRequest"]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    final_df = titles.iloc[movie_indices].head(num)
    final_df.index = range(1, len(final_df) + 1)

    return final_df

def get_movie_details(movie_name):
    movie_row = smd2[smd2['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return f"No information found for '{movie_name}'. Please check the title and try again."
    
    details = []
    for i, (_, row) in enumerate(movie_row.iterrows(), start=1):
        details.append(f"**Movie Reference {i}:**\n")
        details.append(f"**Title:** {row['title']}\n")
        details.append(f"**Overview:** {row['overview']}\n")
        
        directors = set(row['directors'])
        details.append(f"**Director:** {', '.join(directors)}\n")
        
        details.append(f"**Rating:** {row['vote_average']}\n")
        details.append(f"**Release Date:** {row['release_date']}\n")
    
    return "\n".join(details)

# Streamlit App
st.title("🎬 Movie Recommendation Bot ")
st.write("Welcome! This bot will help you discover movies and get details about your favorites.")

# Interface option
option = st.selectbox("What would you like to do?", ("Get Movie Recommendations", "Get Movie Details"))

if option == "Get Movie Recommendations":
    st.subheader("Tell me about your movie preferences:")
    user_input = st.text_area("Enter your preferences (e.g., genre, director, keywords, etc.):")
    
    if st.button("Get Recommendations"):
        if user_input:
            recommendations = getPredictionsV2(user_input, smd2, 10)
            st.write("🎉 Here are some movie recommendations based on your preferences:")
            for idx, title in enumerate(recommendations, start=1):
                st.write(f"{idx}. **{title}**")
        else:
            st.warning("Please enter your preferences.")

elif option == "Get Movie Details":
    st.subheader("Which movie would you like to know about?")
    movie_name = st.text_input("Enter the movie title:")
    
    if st.button("Get Movie Details"):
        if movie_name:
            details = get_movie_details(movie_name)
            st.markdown(details)
        else:
            st.warning("Please enter a movie title.")
