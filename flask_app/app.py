import io
import pickle
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import os
import requests
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

app = Flask(__name__)
CORS(app)


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model(model_path, vectorizer_path):
    """Load the local trained model and vectorizer."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model(
    "../lgbm_model.pkl",
    "../tfidf_vectorizer.pkl"
)

@app.route("/")
def home():
    return "Welcome to the YouTube Sentiment Analysis Flask API"


@app.route("/fetch_comments", methods=["POST"])
def fetch_comments():
    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400

    if not YOUTUBE_API_KEY:
        return jsonify({"error": "YOUTUBE_API_KEY not found in .env"}), 500

    comments = []
    page_token = ""

    try:
        while len(comments) < 500:
            url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "pageToken": page_token,
                "key": YOUTUBE_API_KEY,
            }

            response = requests.get(url, params=params)
            data = response.json()

            if "error" in data:
                return jsonify({"error": data["error"]["message"]}), 500

            if "items" in data:
                for item in data["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "text": snippet["textOriginal"],
                        "timestamp": snippet["publishedAt"],
                        "authorId": snippet.get("authorChannelId", {}).get("value", "Unknown")
                    })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return jsonify(comments)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = " ".join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
