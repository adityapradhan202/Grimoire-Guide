import streamlit as st
from notebooks.recommender import recommend
import time

from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Absolute paths
df = pd.read_csv("D:/projects-aiml/book_recommendation/data/processed.csv")

# Loading the components 
loaded_vec = load(filename="saved_components/vectorizer.pickle")
loaded_sparse = load(filename="saved_components/pdesc_sparse.pickle")
loaded_nlp_en = load(filename="saved_components/en_model_sm.pickle")


st.header("Grimoire Guide🧙‍♂️")
st.write("Grimoire Guide is a recommendation system that leverages data science and machine learning concepts to recommends you the best book based on a plot...")

with st.chat_message(name="Wizard", avatar="assistant"):
    st.write("Describe the plot of a book and I will find the best results for you by using my magical powers. It might take some time because I'm an old wizard :D")


prompt = st.chat_input(placeholder="Describe the plot here...", max_chars=500)
if prompt != None:
    with st.chat_message(name="Wizard", avatar="assistant"):

        st.write("Wait for a few seconds, it might take some time...")
        
    response = recommend(desc=prompt, nlp_model=loaded_nlp_en, vectorizer=loaded_vec, df=df, desc_sparse=loaded_sparse)

    with st.chat_message(name="user"):
        st.write(f"Your prompt: {prompt}")

    if len(response) != 0:
        with st.chat_message(name="Wizard", avatar="assistant"):
            # st.write(response)
            for ind, key in enumerate(response):
                st.markdown(f":orange[{ind+1}. {key}]")
                # st.markdown(f":green[Description:] {response[key]["description"]}")
                st.markdown(f"Author: {response[key]["author"]}, Ratings: {response[key]["avg_rating"]}")
                st.markdown(f"Genres: {response[key]["genres"]}")
                st.markdown(f"URL: {response[key]["url"]}")

            st.markdown("I hope I was helpful to you! Sometimes I might not be able to do the job properly, and I hope that you would forgive me in that case.")
    else:

        st.write("OOPS! Sorry I am not able to find anything related to this plot, please try something else.")


