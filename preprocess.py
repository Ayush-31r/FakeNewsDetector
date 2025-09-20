import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path: str):
    """
    Loads fake/real news dataset.
    Dataset must have 'text' and 'label' columns.
    """
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Splits into train/test sets and vectorizes text.
    """
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer