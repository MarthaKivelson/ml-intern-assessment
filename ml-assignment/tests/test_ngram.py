import pytest
import pandas as pd
from src.ngram_model import TrigramModel
from data.data_preprocessing import preprocess_dataframe, prepare_ngrams

def preprocess_for_model(text):
    """Preprocess raw text exactly like your full pipeline."""
    df = pd.DataFrame([text], columns=["text"])
    df = preprocess_dataframe(df, "text")
    ngram_df = prepare_ngrams(df, "text", n=3)
    return ngram_df["tokens"].tolist()

def test_fit_and_generate():
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    text = preprocess_for_model(text)
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 0

def test_empty_text():
    model = TrigramModel()
    text = ""
    model.fit(text)
    generated_text = model.generate()
    assert generated_text == ""

def test_short_text():
    model = TrigramModel()
    text = "I am."
    model.fit(text)
    generated_text = model.generate()
    assert isinstance(generated_text, str)


