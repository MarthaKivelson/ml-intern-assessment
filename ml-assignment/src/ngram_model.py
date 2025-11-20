import random
import pandas as pd
import ast
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # 1. Data Structure: Nested Dictionary
        # Structure: self.model[(w1, w2)][w3] = count
        # This maps a 'context' (tuple of 2 words) to a Counter of possible next words.
        self.model = defaultdict(Counter)
        
    def fit(self, sentences):
        """
        Trains the trigram model on the given text (list of token lists).

        Args:
            sentences (list of lists): Padded sentences (e.g. from the training CSV).
        """
        print(f"Training on {len(sentences)} sentences...")
        
        for sentence in sentences:
            # Iterate through the sentence to create trigrams
            # We stop at len-2 because we need a lookahead of 3 items
            for i in range(len(sentence) - 2):
                w1 = sentence[i]       # History word 1
                w2 = sentence[i+1]     # History word 2
                w3 = sentence[i+2]     # Target word
                
                # The "Context" is the previous two words
                context = (w1, w2)
                
                # Store the count: counts[w1, w2][w3] += 1
                self.model[context][w3] += 1
                
        print(f"Training complete. Learned {len(self.model)} unique contexts.")

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.
        
        Uses Probabilistic Sampling (not greedy search).
        """
        # 1. Start with the standard padding
        current_sequence = ['<s>', '<s>']
        generated_sentence = []
        
        for _ in range(max_length):
            # Get current context (last two words)
            w1 = current_sequence[-2]
            w2 = current_sequence[-1]
            context = (w1, w2)
            
            # Retrieve possible next words (The Counter object)
            possible_next_words = self.model[context]
            
            # Dead End Check: If the model has never seen this sequence of 2 words, stop.
            if not possible_next_words:
                break 
            
            # --- THE PROBABILISTIC LOGIC ---
            
            # 1. Get the candidates and their raw counts
            words = list(possible_next_words.keys())
            counts = list(possible_next_words.values())
            
            # 2. Convert Counts to Probabilities (Conceptually)
            # random.choices handles the math: P(w) = count(w) / sum(all_counts)
            # weights=counts tells Python to pick items with higher counts more often.
            
            next_word = random.choices(words, weights=counts, k=1)[0]
            
            # -------------------------------

            # Stop if we hit the end token
            if next_word == '</s>':
                break
                
            # Append to result and update sequence for next iteration
            generated_sentence.append(next_word)
            current_sequence.append(next_word)
            
        return " ".join(generated_sentence)

def main():
    print("Loading Data...")
    
    # Load the CSV we created in data_preprocessing.py
    # IMPORTANT: 'converters' is needed to turn the string "['a','b']" back into a real list
    try:
        ngram_df = pd.read_csv(
            "./data/interim/training_data.csv", 
            converters={'tokens': ast.literal_eval}
        )
    except FileNotFoundError:
        print("Error: Training data not found. Run data_preprocessing.py first.")
        return

    # Initialize
    model = TrigramModel()
    
    # Train
    # Pass the Series of lists to the fit method
    model.fit(ngram_df['tokens'])
    
    # Generate
    print("\n" + "="*40)
    print("GENERATED TEXT SAMPLES:")
    print("="*40)
    for i in range(10):
        text = model.generate()
        print(f"{i+1}. {text}")
    print("="*40)

if __name__ == '__main__':
    main()