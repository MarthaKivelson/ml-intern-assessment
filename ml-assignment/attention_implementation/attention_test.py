import numpy as np
import pytest
from attention_task import scaled_dot_product_attention, softmax

# Set a seed for reproducibility
np.random.seed(42)

def test_softmax():
    """
    Test if softmax output sums to 1 along the last axis.
    """
    x = np.random.rand(2, 5)
    s = softmax(x, axis=-1)
    
    # Check if sum is close to 1
    assert np.allclose(np.sum(s, axis=-1), 1.0), "Softmax probabilities should sum to 1"
    
    # Check if values are in range [0, 1]
    assert np.all((s >= 0) & (s <= 1)), "Softmax output should be between 0 and 1"

def test_attention_shapes():
    """
    Test input/output shapes of the attention function.
    """
    batch_size = 2
    seq_len_q = 5
    seq_len_k = 6  # Key/Value sequence length
    d_k = 8        # Dimension of keys/queries
    d_v = 10       # Dimension of values
    
    Q = np.random.rand(batch_size, seq_len_q, d_k)
    K = np.random.rand(batch_size, seq_len_k, d_k)
    V = np.random.rand(batch_size, seq_len_k, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Expected shapes
    assert output.shape == (batch_size, seq_len_q, d_v), f"Expected output shape {(batch_size, seq_len_q, d_v)}, got {output.shape}"
    assert weights.shape == (batch_size, seq_len_q, seq_len_k), f"Expected weights shape {(batch_size, seq_len_q, seq_len_k)}, got {weights.shape}"

def test_attention_masking():
    """
    Test if masking correctly zeroes out attention weights.
    """
    batch_size = 1
    seq_len_q = 1
    seq_len_k = 3
    d_k = 4
    d_v = 4
    
    Q = np.random.rand(batch_size, seq_len_q, d_k)
    K = np.random.rand(batch_size, seq_len_k, d_k)
    V = np.random.rand(batch_size, seq_len_k, d_v)
    
    # Mask out the last token (index 2)
    # Mask shape: (batch, seq_len_q, seq_len_k)
    mask = np.array([[[1, 1, 0]]]) 
    
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # The weight for the masked position (last element) should be 0 (or extremely close to it)
    masked_weight = weights[0, 0, 2]
    
    assert np.isclose(masked_weight, 0.0, atol=1e-7), f"Masked position weight should be 0, got {masked_weight}"
    
    # The weights for unmasked positions should still sum to 1
    assert np.isclose(np.sum(weights), 1.0), "Attention weights should still sum to 1 after masking"

def test_attention_values():
    """
    Test attention logic with a trivial example where Q matches one K perfectly.
    """
    # 1 Query, 2 Keys, dim=4
    # Q matches K1 exactly, and is orthogonal to K2
    Q = np.array([[[1, 0, 0, 0]]]) # Shape (1, 1, 4)
    
    K = np.array([[
        [10, 0, 0, 0], # K1 (Strong match)
        [0, 10, 0, 0]  # K2 (No match/Orthogonal)
    ]]) # Shape (1, 2, 4)
    
    V = np.array([[
        [1, 1, 1, 1], # V1
        [2, 2, 2, 2]  # V2
    ]]) # Shape (1, 2, 4)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Since Q aligns perfectly with K1 and not K2, 
    # the weight for K1 should be much higher than K2.
    w1 = weights[0, 0, 0]
    w2 = weights[0, 0, 1]
    
    assert w1 > w2, "Query should attend more to the matching Key"
    
    # The output should be closer to V1 than V2
    # Since w1 is high, output should be closer to [1, 1, 1, 1]
    assert np.all(output < 1.5), "Output should be dominated by V1"

if __name__ == "__main__":
    # Allow running this script directly to see passes
    import sys
    try:
        test_softmax()
        test_attention_shapes()
        test_attention_masking()
        test_attention_values()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)