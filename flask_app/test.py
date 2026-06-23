import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    comment = comment.lower().strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
    stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
    comment = " ".join([word for word in comment.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])
    return comment


def load_model(model_path, vectorizer_path):
    """Load the trained model and vectorizer locally."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model(
    "../lgbm_model.pkl",
    "../tfidf_vectorizer.pkl"
)

def predict():
    comments = ["I love this video!", "This is not a good explanation."]

    preprocessed_comments = [preprocess_comment(comment) for comment in comments]
    transformed_comments = vectorizer.transform(preprocessed_comments)
    dense_comments = transformed_comments.toarray()
    predictions = model.predict(dense_comments).tolist()

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    print(response)


if __name__ == "__main__":
    predict()
