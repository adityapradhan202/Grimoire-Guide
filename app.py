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
# these are just for annotation and are never called.
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_resource
def loading_components():
    """Load pre-trained components and processed dataset.

    :return: A tuple containing vectorizer, sparse_matrix, nlp_model, df
    """
    vectorizer = load(filename="./saved_components/vectorizer.pickle")
    sparse_matrix = load(filename="./saved_components/pdesc_sparse.pickle")
    nlp_model = load(filename="./saved_components/en_model_sm.pickle")
    dataframe = pd.read_csv("./data/processed.csv")

    return vectorizer, sparse_matrix, nlp_model, dataframe


loaded_vec, loaded_sparse, loaded_nlp_en, df = loading_components()


def process_description(
        description: str,
        nlp_model: spacy.language.Language = loaded_nlp_en
) -> str:
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


def find_similar_description(
        description: str,
        nlp_model: spacy.language.Language = loaded_nlp_en,
        vectorizer: TfidfVectorizer = loaded_vec,
        desc_sparse: csr_matrix = loaded_sparse
):
    """Finds indices of descriptions similar to the given description based on cosine similarity.

    :param description: The input description to compare.
    :type description: Str
    :param nlp_model: A loaded spaCy NLP model for text processing.
    :type nlp_model: spacy.language.Language
    :param vectorizer: A trained TF-IDF vectorizer for text transformation.
    :type vectorizer: TfidfVectorizer
    :param desc_sparse: Precomputed sparse matrix of descriptions.
    :return:
    """
    processed_text = process_description(description=description,
                                         nlp_model=nlp_model)
    processed_text_vector = vectorizer.transform([processed_text])
    # similarity_scores is a numpy array
    similarity_scores = cosine_similarity(processed_text_vector, desc_sparse)

    # We start with a high similarity threshold and keep lowering it.
    # This way, we first try to find very close matches.
    # If no match is found, we relax the condition and check again.
    # If we find any match at any step,
    # we return immediately, no need to check further.
    for thresh in [0.4, 0.3, 0.2, 0.1]:
        matching_indices = np.where((similarity_scores >= thresh)
                                    & (similarity_scores < 1.0))[0].tolist()
        # If we get matches, we return them immediately.
        if matching_indices :
            return matching_indices


def collect_book_data(
        matching_indices: list[int],
        df: pd.DataFrame
) -> dict[str, dict[str,str]]:
    """Collects book details for the given matching indices.

    :param matching_indices: List of indices of similar books.
    :param df: The dataset contains book details.
    :return: A dictionary where the book title is the key,
        and the value is another dictionary
        containing book details like description,
        author, genres, average rating, and URL.
    """
    book_details= {}
    for index in matching_indices:
        book_details[df["Book"].iloc[index]] = {
            "description": df["Description"].iloc[index],
            "author": df["Author"].iloc[index],
            "genres": df["Genres"].iloc[index],
            "avg_rating": df["Avg_Rating"].iloc[index],
            "url": df["URL"][index]
        }

    return book_details


def recommend_books(
        description: str,
        df: pd.DataFrame,
        nlp_model: spacy.language.Language = loaded_nlp_en,
        vectorizer: TfidfVectorizer = loaded_vec,
        desc_sparse: csr_matrix =loaded_sparse,
) -> dict[str, dict[str, str]]:
    """Recommends books similar to the given description.

    :param description: The input book description.
    :type description: Str
    :param nlp_model: The NLP model for text processing.
    :type nlp_model: spacy.language.Language
    :param vectorizer: The trained TF-IDF vectorizer.
    :type vectorizer: TfidfVectorizer
    :param desc_sparse: The sparse matrix of book descriptions.
    :type desc_sparse: Csr_matrix
    :param df: The dataset contains book details.
    :type df: pd.DataFrame
    :return: A dictionary where the book title is the key,
        and the value is another dictionary containing book details like description,
        author, genres, average rating, and URL.
    :rtype: dict[str, dict[str, str]]
    """
    matched_inds = find_similar_description(description, nlp_model=nlp_model,
                                            vectorizer=vectorizer,
                                            desc_sparse=desc_sparse)
    res = collect_book_data(matching_indices=matched_inds, df=df)
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

    response = recommend_books(description=prompt, nlp_model=loaded_nlp_en,
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