from transformers import pipeline


def perform_sentiment_analysis(text):
    classifier = pipeline("sentiment-analysis")
    sentiment = classifier(text)[0]
    return sentiment['label']