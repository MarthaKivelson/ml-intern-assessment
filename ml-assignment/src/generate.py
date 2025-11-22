from ngram_model import TrigramModel
from data.data_preprocessing import preprocess_dataframe, prepare_ngrams
import pandas as pd

def preprocess_for_model(text):
    """Preprocess raw text exactly like your full pipeline."""
    df = pd.DataFrame([text], columns=["text"])
    # step 1: clean
    df = preprocess_dataframe(df, "text")
    # step 2: tokenize + pad + UNK replacement
    ngram_df = prepare_ngrams(df, "text", n=3)
    # return the list-of-tokens (not dataframe)
    return ngram_df["tokens"].tolist()


def main():
    # Create a new TrigramModel
    model = TrigramModel()

    # Train the model on the example corpus
    with open("ml-intern-assessment\FRANKENSTEIN.txt", "r") as f:
        text = f.read()
    
    text = preprocess_for_model(text)
    model.fit(text)

    # Generate new text
    generated_text = model.generate()
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
