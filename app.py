"""
app.py
This is streamlit app runner script

@author: adityapradhan202, Asifdotexe
Last modified: 04-03-2025
"""

import time
import numpy as np
import pandas as pd
import spacy.language
import streamlit as st
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def loading_components():
    """Load pre-trained components and processed dataset.

    :return: A tuple containing vectorizer, sparse_matrix, nlp_model, df
    """
    vectorizer = load(filename="./saved_components/vectorizer.pickle")
    sparse_matrix = load(filename="./saved_components/pdesc_sparse.pickle")
    nlp_model = load(filename="./saved_components/en_model_sm.pickle")
    df = pd.read_csv("./data/processed.csv")

    return vectorizer, sparse_matrix, nlp_model, df


loaded_vec, loaded_sparse, loaded_nlp_en, df = loading_components()


def process_description(description: str,
                        nlp_model: spacy.language.Language=loaded_nlp_en) -> str:
    """Processes a text description by removing stopwords and punctuation,
    then lemmatizing the remaining tokens.

    :param description: The text description to be processed.
    :type description: Str
    :param nlp_model: A loaded spaCy NLP model.
    :type nlp_model: spacy.language.Language
    :return: The cleaned and lemmatized description.
    :rtype: Str
    """
    doc = nlp_model(description)
    filtered_tokens = [token.lemma_ for token in doc
                       if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)


def similar_description(desc, nlp_model=loaded_nlp_en,
                        vectorizer=loaded_vec, desc_sparse=loaded_sparse):
    ptext = process_description(description=desc, nlp_model=nlp_model)
    ptext_vec = vectorizer.transform([ptext])
    csim = cosine_similarity(ptext_vec, desc_sparse)
    # csim is a numpy array
    # iteration over it will be very fast

    matched_inds = []
    for thresh in [0.4, 0.3, 0.2, 0.1]:
        matched_inds = np.where((csim >= thresh) & (csim < 1.0))[
            0].tolist()
        if matched_inds:
            return matched_inds


def collect_data(matched_inds, df=df):
    res_dict = {}
    for ind in matched_inds:
        res_dict[df["Book"].iloc[ind]] = {
            "description": df["Description"].iloc[ind],
            "author": df["Author"].iloc[ind],
            "genres": df["Genres"].iloc[ind],
            "avg_rating": df["Avg_Rating"].iloc[ind],
            "url": df["URL"][ind]
        }

    return res_dict


def recommend(desc, nlp_model=loaded_nlp_en,
              vectorizer=loaded_vec, desc_sparse=loaded_sparse, df=df):
    matched_inds = similar_description(desc, nlp_model=nlp_model,
                                       vectorizer=vectorizer,
                                       desc_sparse=desc_sparse)
    res = collect_data(matched_inds=matched_inds, df=df)
    return res


# app code
st.header("Grimoire Guide 🧙‍♂️")
st.write(
    "Grimoire Guide is a recommendation system that leverages data science and machine learning concepts to recommend you the best boosk based on a plot or a description...")

with st.chat_message(name="Wizard", avatar="assistant"):
    st.write(
        "Describe the plot of a book and I will find the best results for you by using my magical powers. It might take some time because I'm an old wizard :D")

prompt = st.chat_input(placeholder="Describe the plot here...", max_chars=500)
if prompt != None:
    start_time = time.time()  # Start timing
    with st.chat_message(name="Wizard", avatar="assistant"):

        st.write("Wait for a few seconds, it might take some time...")

    response = recommend(desc=prompt, nlp_model=loaded_nlp_en,
                         vectorizer=loaded_vec, df=df,
                         desc_sparse=loaded_sparse)

    end_time = time.time()  # End timing
    response_time = round(end_time - start_time,2)  # Calculate response time in seconds

    with st.chat_message(name="user"):
        st.write(f"Your prompt: {prompt}")

    if len(response) != 0:
        with st.chat_message(name="Wizard", avatar="assistant"):

            for ind, key in enumerate(response):
                st.markdown(f":orange[{ind + 1}. {key}]")

                st.markdown(
                    f"Author: {response[key]["author"]}, Ratings: {response[key]["avg_rating"]}")
                st.markdown(f"Genres: {response[key]["genres"]}")
                st.markdown(f"URL: {response[key]["url"]}")

            st.markdown(f"I hope I was helpful to you! Sometimes I might not be able to do the job properly, and I hope that you would forgive me in that case. \n\n Response time: **{response_time} seconds**.")
    else:

        st.write(
            "OOPS! Sorry I am not able to find anything related to this plot, please try something else.")