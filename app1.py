# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
pd.pandas.set_option('display.max_columns', None)

movies_data = pd.read_csv('./data/movies_metadata.csv')
link_small = pd.read_csv('./data/links_small.csv')
credits = pd.read_csv('./data/credits.csv')
keyword = pd.read_csv('./data/keywords.csv')

# Removing rows with the index labels 19730, 29503, and 35587
movies_data = movies_data.drop([19730, 29503, 35587])

# Convert the 'id' column to integers
movies_data['id'] = movies_data['id'].astype('int')

# Filtering 'link_small' DataFrame to get rows where 'tmdbId' is not null and converting 'tmdbId' to integers
link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# Creating 'smd' DataFrame by filtering 'movies_data' based on 'id' values present in 'link_small'
smd = movies_data[movies_data['id'].isin(link_small)]

# Filling missing values in 'tagline' and 'overview' columns with empty strings
smd['tagline'] = smd['tagline'].fillna('')
smd['overview'] = smd['overview'].fillna('')

keyword['id'] = keyword['id'].astype('int')
credits['id'] = credits['id'].astype('int')

movies_data_merged = movies_data.merge(keyword, on='id')
movies_data_merged = movies_data_merged.merge(credits, on='id')

smd2 = movies_data_merged[movies_data_merged['id'].isin(link_small)]

smd2['cast'] = smd2['cast'].apply(literal_eval)
smd2['crew'] = smd2['crew'].apply(literal_eval)
smd2['keywords'] = smd2['keywords'].apply(literal_eval)
smd2['cast_size'] = smd2['cast'].apply(lambda x: len(x))
smd2['crew_size'] = smd2['crew'].apply(lambda x: len(x))

# Extracting description from tagline and overview features
smd2['tagline'] = smd2['tagline'].fillna('').apply(lambda x: x.split(" "))
smd2['overview'] = smd2['overview'].fillna('').apply(lambda x: x.split(" "))
smd2["description"] = smd2['tagline'] + smd2['overview']

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
test1["soup"] = smd2['keywords'] + smd2['cast'] + smd2['directors'] + smd2["description"]
test1['soup'] = test1['soup'].apply(lambda x: " ".join(x))
test1['soup'] = test1['soup'].apply(lambda x: remove_stopwords(x))

stemmer = SnowballStemmer('english')
test1['soup'] = test1['soup'].apply(lambda x: x.split(" ")).apply(lambda x: [stemmer.stem(i) for i in x])
test1['soup'] = test1['soup'].apply(lambda x: " ".join(x))

d = {}
for item in ['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
             'imdb_id', 'original_language', 'original_title', 'overview',
             'popularity', 'poster_path', 'production_companies',
             'production_countries', 'release_date', 'revenue', 'runtime',
             'spoken_languages', 'status', 'tagline', 'title', 'video',
             'vote_average', 'vote_count', 'keywords', 'cast', 'crew', 'cast_size',
             'crew_size', 'description', 'directors', 'soup']:
    d[item] = "NA"

# ## **getPredictions Function**
def getPredictionsV2(soup, smd, num):
    smd = smd[smd.title != "myRequest"]

    # Remove stopwords
    stopword_removed_soup = remove_stopwords(soup)

    # Stem the input string
    stemmer = SnowballStemmer('english')
    soup_list = stopword_removed_soup.split(" ")
    soup_list_stemmed = [stemmer.stem(i) for i in soup_list]
    stemmed_soup = " ".join(soup_list_stemmed)

    d["soup"] = stemmed_soup
    d["title"] = "myRequest"
    smd = pd.concat([smd, pd.DataFrame([d])], ignore_index=True)

    # Fill NaN values in 'soup' with empty strings
    smd['soup'] = smd['soup'].fillna('')

    # Create TF-IDF matrix
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfid_mat = tf.fit_transform(smd['soup'])
    cos_sim = linear_kernel(tfid_mat, tfid_mat)

    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])

    idx = indices["myRequest"]  # Getting the index of the movie
    sim_scores = list(enumerate(cos_sim[idx]))  # Finding the cos similarity of the movie using its index
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sorting the movie based on the similarity score
    sim_scores = sim_scores[1:31]  # Taking first 30 movies
    movie_indices = [i[0] for i in sim_scores]  # Taking the sorted movies

    final_df = titles.iloc[movie_indices].head(num)
    final_df.index = range(1, len(final_df) + 1)

    # Use str.join() to concatenate index and movie names with the desired separator
    formatted_output = "\n".join(f"{i}. {movie}" for i, movie in enumerate(final_df, start=1))

    return formatted_output

# Change the way you store and format the overview and director's name in the get_movie_details function
def get_movie_details(movie_name):
    # Searching for the movie in the dataset
    movie_row = smd2[smd2['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return f"No information found for '{movie_name}'. Please check the title and try again."
    
    details = []
    for i, (_, row) in enumerate(movie_row.iterrows(), start=1):  # Start enumeration at 1
        details.append(f"**Movie Reference {i}:**\n")  # Add numbering
        details.append(f"**Title:** {row['title']}\n")
        
        # Join overview list into a single string for better formatting
        details.append(f"**Overview:** {' '.join(row['overview'])}\n")
        
        # Ensure the director's name appears only once
        directors = set(row['directors'])  # Use a set to avoid duplicates
        details.append(f"**Director:** {', '.join(directors)}\n")
        
        details.append(f"**Rating:** {row['vote_average']}\n")
        details.append(f"**Release Date:** {row['release_date']}\n")
    
    # Add spacing for better readability
    return "\n".join(details)

# Update this part to keep the existing code intact


# Building streamlit app
st.title("Movie Recommendation Bot")
with st.chat_message("assistant"):
    st.write("""Welcome to our Movie Recommendation Chatbot!

I'm your friendly movie chatbot, and I'm here to help you discover your next favorite film. 
You can either get recommendations based on your preferences or find details about a specific movie.""")

# Option selection for recommendations or movie details
option = st.selectbox("What would you like to do?", ("Get Movie Recommendations", "Get Movie Details"))

if option == "Get Movie Recommendations":
    st.subheader("Tell me about your movie preferences!")
    user_input = st.text_input("Enter your preferences (genre, director, keywords, etc.)")
    
    if st.button("Get Recommendations"):
        if user_input:
            recommendations = getPredictionsV2(user_input, smd2, 10)
            st.write("Here are some movie recommendations based on your preferences:")
            st.write(recommendations)
        else:
            st.warning("Please enter your preferences.")

elif option == "Get Movie Details":
    st.subheader("Which movie would you like to know about?")
    movie_name = st.text_input("Enter the movie title")
    
    if st.button("Get Movie Details"):
        if movie_name:
            details = get_movie_details(movie_name)
            st.write(details)
        else:
            st.warning("Please enter a movie title.")