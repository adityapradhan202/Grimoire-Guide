from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Absolute paths
df = pd.read_csv("data/processed.csv")

# Loading the components
loaded_vec = load(filename="D:/projects-aiml/book_recommendation/saved_components/vectorizer.pickle")
loaded_sparse = load(filename="D:/projects-aiml/book_recommendation/saved_components/pdesc_sparse.pickle")
loaded_nlp_en = load(filename="D:/projects-aiml/book_recommendation/saved_components/en_model_sm.pickle")


def process_description(desc, nlp_model=loaded_nlp_en):
    doc = nlp_model(desc)
    filtered = []
    for token in doc:
        if (not token.is_stop) and (not token.is_punct):
            filtered.append(token.lemma_)

    return " ".join(filtered)

def similar_description(desc, nlp_model=loaded_nlp_en, 
                        vectorizer=loaded_vec, desc_sparse=loaded_sparse):
    
    matched_inds = []
    thresh_range = [0.3, 0.2, 0.1]
    for thresh in thresh_range:

        ptext = process_description(desc=desc, nlp_model=nlp_model)
        ptext_vec = vectorizer.transform([ptext])

        for ind, desc_sp in enumerate(desc_sparse):
            csim = cosine_similarity(ptext_vec, desc_sp)
            if csim[0][0] >= thresh and csim < 1.0:
                matched_inds.append(ind)

        if len(matched_inds) != 0:
            return matched_inds
        
def collect_data(matched_inds, df=df):
    res_dict = {}
    for ind in matched_inds:
        res_dict[df["Book"].iloc[ind]] = {
            "description":df["Description"].iloc[ind],
            "author":df["Author"].iloc[ind],
            "genres":df["Genres"].iloc[ind],
            "avg_rating":df["Avg_Rating"].iloc[ind],
            "url":df["URL"][ind]
        }

    return res_dict

def recommend(desc, nlp_model=loaded_nlp_en, 
                        vectorizer=loaded_vec, desc_sparse=loaded_sparse, df=df):
    
    matched_inds = similar_description(desc, nlp_model=nlp_model, 
                        vectorizer=vectorizer, desc_sparse=desc_sparse)
    res = collect_data(matched_inds=matched_inds, df=df)
    return res
    
if __name__ == "__main__":
    res = recommend(
        desc="A soldier killed so may enemies in the war and fought bravely!"
    )
    print(res)