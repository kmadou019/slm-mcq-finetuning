import re
import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))

def calculate_originality_for_df(df: pd.DataFrame,
                                 originality_col: str,
                                 question_col: str,
                                 lisa_sheet_col: str):

    def clean_text(text):
        text = re.sub(r'\W+', ' ', text).lower().strip()

        words = word_tokenize(text)

        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return cleaned_words

    def get_trigrams(text):
        words = clean_text(text)
        return set(ngrams(words, 3))

    def calculate_originality(question, lisa_sheet):
        question_trigrams = get_trigrams(question)
        lisa_sheet_trigrams = get_trigrams(lisa_sheet)

        unique_trigrams = question_trigrams - lisa_sheet_trigrams
        originality_score = len(unique_trigrams) / len(question_trigrams) if question_trigrams else 0
        return originality_score
    df[originality_col] = df.apply(lambda row: calculate_originality(row[question_col], row[lisa_sheet_col]), axis=1)
    return df