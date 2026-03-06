import re

import pandas as pd


def calculate_readability_for_df(df: pd.DataFrame,
                                 readability_col: str,
                                 question_col: str):
    # Fleish-Kincaid algorithm
    def syllable_count(word):
        word = word.lower()
        syllable_count = len(re.findall(r'[aeiouy]+', word))
        return max(1, syllable_count)

    def compute_readability(text):
        words = text.split()
        num_words = len(words)
        num_sentences = text.count('.') + text.count('!') + text.count('?')
        num_syllables = sum(syllable_count(word) for word in words)

        if num_words == 0 or num_sentences == 0:
            return None  # To avoid division by zero

        readability = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        return readability

    df[readability_col] = df[question_col].apply(compute_readability)
    return df