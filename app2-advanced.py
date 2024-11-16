# Import libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
from ast import literal_eval

# Load and preprocess data only once
@st.cache_data
def load_and_preprocess_data():
    # Load datasets
    movies_data = pd.read_csv('./data/movies_metadata.csv')
    link_small = pd.read_csv('./data/links_small.csv')
    credits = pd.read_csv('./data/credits.csv')
    keyword = pd.read_csv('./data/keywords.csv')

    # Data preprocessing
    movies_data = movies_data.drop([19730, 29503, 35587])
    movies_data['id'] = movies_data['id'].astype('int')
    link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    
    smd = movies_data[movies_data['id'].isin(link_small)]
    smd['tagline'] = smd['tagline'].fillna('')
    smd['overview'] = smd['overview'].fillna('')
    
    # Merging
    movies_data_merged = movies_data.merge(keyword, on='id').merge(credits, on='id')
    smd2 = movies_data_merged[movies_data_merged['id'].isin(link_small)]
    smd2['cast'] = smd2['cast'].apply(literal_eval).apply(lambda x: [i['name'] for i in x][:3] if isinstance(x, list) else [])
    smd2['crew'] = smd2['crew'].apply(literal_eval)
    smd2['keywords'] = smd2['keywords'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    # Get directors
    def get_director(crew):
        for i in crew:
            if i.get('job') == 'Director':
                return i['name']
        return ""
    
    smd2['directors'] = smd2['crew'].apply(get_director).apply(lambda x: [x, x, x] if x else [])
    smd2['overview'] = smd2['overview'].apply(lambda x: [x] if isinstance(x, str) else x)
    
    # Create "soup"
    smd2["soup"] = smd2['keywords'] + smd2['cast'] + smd2['directors'] + smd2['overview']
    smd2['soup'] = smd2['soup'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    smd2['soup'] = smd2['soup'].apply(lambda x: remove_stopwords(x)).apply(lambda x: " ".join(SnowballStemmer('english').stem(word) for word in x.split()))
    
    return smd2

smd2 = load_and_preprocess_data()

@st.cache_data
def get_tfidf_matrix(smd):
    # Vectorize once and cache the matrix
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['soup'])
    return tfidf_matrix, tf

tfidf_matrix, tf = get_tfidf_matrix(smd2)

# Prediction function
def getPredictionsV2(user_input, smd, num):
    # Create a single-row DataFrame for the input
    stopword_removed_soup = remove_stopwords(user_input)
    stemmed_soup = " ".join(SnowballStemmer('english').stem(word) for word in stopword_removed_soup.split())
    
    input_tfidf = tf.transform([stemmed_soup])
    cosine_sim = linear_kernel(input_tfidf, tfidf_matrix).flatten()
    
    # Get the top N movie indices based on similarity
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    return smd.iloc[movie_indices][['title']]

# Movie details function
def get_movie_details(movie_name):
    movie_row = smd2[smd2['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return f"No information found for '{movie_name}'. Please check the title and try again."
    
    details = []
    for i, (_, row) in enumerate(movie_row.iterrows(), start=1):
        details.append(f"**Movie Reference {i}:**\n")
        details.append(f"**Title:** {row['title']}\n")
        details.append(f"**Overview:** {row['overview']}\n")
        details.append(f"**Director:** {', '.join(set(row['directors']))}\n")
        details.append(f"**Rating:** {row['vote_average']}\n")
        details.append(f"**Release Date:** {row['release_date']}\n")
    
    return "\n".join(details)

# Streamlit App Interface
# Streamlit App Interface
st.title("🎬 Movie Recommendation Bot")
st.write("Welcome! This bot will help you discover movies and get details about your favorites.")

option = st.selectbox("What would you like to do?", ("Get Movie Recommendations", "Get Movie Details"))

if option == "Get Movie Recommendations":
    st.subheader("Tell me about your movie preferences:")
    user_input = st.text_area("Enter your preferences (e.g., genre, director, keywords, etc.):")
    
    if st.button("Get Recommendations"):
        if user_input:
            recommendations = getPredictionsV2(user_input, smd2, 10)
            st.write("🎉 Here are some movie recommendations based on your preferences:")
            for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
                st.write(f"{i}. **{row['title']}**")
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
