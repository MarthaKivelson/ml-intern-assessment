import nltk
import re
import pandas as pd
import os
import string
from collections import Counter
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('stopwords')

def preprocess_dataframe(df, col='text'):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Initialize lemmatizer and stopwords
    #lemmatizer = WordNetLemmatizer()
    #stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        # Remove stop words
        #text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        #text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    
    # Apply preprocessing to the specified column
    df[col] = df[col].apply(preprocess_text)
    
    # Remove small sentences (less than 3 words)
    # df[col] = df[col].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)

    # Drop rows with NaN values
    df = df.dropna(subset=[col])
    return df

def prepare_ngrams(df, col='text', n=3):
    """
    Takes the preprocessed DataFrame and creates padded token lists.
    Also handles Unknown Words (<UNK>).
    """
    tokenized_sentences = []
    all_words = []

    # First pass: Tokenize and collect all words to build vocabulary
    temp_tokenized = []
    for text in df[col]:
        tokens = text.split()
        if not tokens: continue
        temp_tokenized.append(tokens)
        all_words.extend(tokens)

    # Build Vocabulary
    # If a word appears only once, we treat it as unknown to help the model generalize
    word_counts = Counter(all_words)
    vocab = {word for word, count in word_counts.items() if count > 1}
    
    final_sentences = []
    
    for tokens in temp_tokenized:
        # Padding for Trigrams (N=3)
        # We need (N-1) start tokens.
        # We use '<s>' for start and '</s>' for end.
        padding_start = ['<s>'] * (n - 1)
        padding_end = ['</s>']
        
        # Handle Unknown Words
        processed_tokens = []
        for token in tokens:
            if token in vocab:
                processed_tokens.append(token)
            else:
                processed_tokens.append('<UNK>')
        
        # Combine Padding + Processed Tokens
        full_sentence = padding_start + processed_tokens + padding_end
        final_sentences.append(full_sentence)
        
    return pd.DataFrame({'tokens': final_sentences})



def main():
    try:

        with open('FRANKENSTEIN.txt', 'r', encoding='utf-8') as f:
            # Read lines and remove empty lines if you want
            lines = [line.strip() for line in f if line.strip()]

        df = pd.DataFrame(lines, columns=['text'])
        print(df.head())
        #logging.info('data loaded properly')

        # Transform the data
        processed_data = preprocess_dataframe(df, 'text')

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        processed_data.to_csv(os.path.join(data_path, "processed.csv"), index=False)

        print("Preparing N-Grams (Padding & Handling Unknowns)...")
        training_data = prepare_ngrams(processed_data, 'text', n=3)
        training_data.to_csv(os.path.join(data_path, "training_data.csv"), index=False)
        print(f"Sample padded sentence: {training_data.head()}")

        
        #logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        #logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()