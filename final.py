import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
from ast import literal_eval
import requests
import time

# TMDB API key
TMDB_API_KEY = 'aad48407a8c1adecea9cc23891d3181a'  # Replace with your actual TMDB API key

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

def get_movie_poster(movie_id, retries=3):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    for i in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return None  # Return None if no poster found
        except requests.exceptions.RequestException as e:
            # Print the exception for debugging (can also log this)
            print(f"Attempt {i + 1}: Connection error occurred: {e}")
            # Wait for a short period before retrying
            time.sleep(2 ** i)  # Exponential backoff
    return None  # Return None if all retries fail

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
    
    return smd.iloc[movie_indices][['title', 'id']]

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
        
        # Fetch and display the movie poster using TMDB API
        poster_url = get_movie_poster(row['id'])
        if poster_url:
            st.image(poster_url, width=200)  # Adjust width as needed
        else:
            st.write("No poster available.")
        details.append("\n")  # Add spacing after each movie poster

    return "\n".join(details)

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

            # Display the posters and titles in 2 rows of 5 columns
            for i in range(0, len(recommendations), 5):
                cols = st.columns(5)
                for j, (_, row) in enumerate(recommendations.iloc[i:i+5].iterrows()):
                    with cols[j]:
                        poster_url = get_movie_poster(row['id'])
                        if poster_url:
                            st.image(poster_url, width=150, use_column_width='always', caption=row['title'])
                        else:
                            st.write("No poster available.")
                        st.markdown("""
                            <style>
                            .movie-poster {
                                border: 5px solid #000000;
                                border-radius: 5px;
                                padding: 10px;
                                margin: 10px;
                                display: inline-block;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        # st.markdown(f'<div class="movie-poster"></div>', unsafe_allow_html=True)
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
