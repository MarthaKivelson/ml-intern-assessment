import numpy as np

def softmax(x, axis=-1):
    """
    Compute the softmax of vector x in a numerically stable way.
    
    Args:
        x (np.array): Input array.
        axis (int): Axis along which to compute softmax.
        
    Returns:
        np.array: Softmax output of the same shape as x.
    """
    # Subtract max for numerical stability (prevents overflow with exp)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculate the Scaled Dot-Product Attention.
    
    Formula: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    
    Args:
        Q (np.array): Query matrix of shape (batch_size, seq_len_q, d_k)
        K (np.array): Key matrix of shape (batch_size, seq_len_k, d_k)
        V (np.array): Value matrix of shape (batch_size, seq_len_v, d_v)
                      Note: In simple implementation seq_len_k usually equals seq_len_v.
        mask (np.array, optional): Mask array. Elements with 1 (or True) are KEPT, 
                                   elements with 0 (or False) are MASKED OUT (set to -inf).
                                   Shape should be broadcastable to (batch_size, seq_len_q, seq_len_k).
                                   
    Returns:
        output (np.array): The attended output of shape (batch_size, seq_len_q, d_v).
        attention_weights (np.array): The attention weights (after softmax) 
                                      of shape (batch_size, seq_len_q, seq_len_k).
    """
    
    # 1. Dimensions
    # d_k is the dimensionality of the keys (and queries).
    # We get it from the last dimension of K.
    d_k = K.shape[-1]
    
    # 2. Score Calculation (Dot Product)
    # We perform matrix multiplication between Q and K transposed.
    # Q shape: (..., seq_len_q, d_k)
    # K.T shape (conceptually): (..., d_k, seq_len_k)
    # Resulting scores shape: (..., seq_len_q, seq_len_k)
    
    # Note on numpy matmul: specifically supports stacks of matrices (batches).
    # We transpose only the last two dimensions of K for the dot product.
    scores = np.matmul(Q, K.swapaxes(-2, -1))
    
    # 3. Scaling
    # We divide by the square root of the dimension of the keys.
    # This prevents the dot products from growing too large in magnitude, 
    # which pushes the softmax function into regions where it has extremely small gradients.
    scores = scores / np.sqrt(d_k)
    
    # 4. Masking (Optional)
    # If a mask is provided, we set the masked positions to a very large negative number (-inf).
    # When softmax is applied, exp(-inf) becomes 0, effectively ignoring these positions.
    if mask is not None:
        # Assuming mask has 0 for positions to mask out and 1 for valid positions.
        # We use a large negative number like -1e9 to simulate -infinity.
        scores = np.where(mask == 0, -1e9, scores)
        
    # 5. Softmax
    # Apply softmax to the last axis (seq_len_k) to obtain attention weights.
    # The weights represent how much focus each query should put on each key.
    # They sum to 1 along the last axis.
    attention_weights = softmax(scores, axis=-1)
    
    # 6. Weighted Sum
    # Multiply the attention weights by the Value matrix V.
    # Weights shape: (..., seq_len_q, seq_len_k)
    # V shape: (..., seq_len_v, d_v)  (Note: seq_len_k usually == seq_len_v)
    # Output shape: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# ==========================================
# Demonstration
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    print("-" * 40)
    print("SCALED DOT-PRODUCT ATTENTION DEMO")
    print("-" * 40)

    # --- Setup Sample Data ---
    # Let's assume:
    # Batch Size = 1 (processing one sequence)
    # Sequence Length = 3 (e.g., "I love AI")
    # d_k (dimension of keys/queries) = 4
    # d_v (dimension of values) = 4 (often same as d_k, but doesn't have to be)
    
    # Sample Queries (Q)
    Q = np.array([
        [1, 0, 1, 0],  # Word 1 Query
        [0, 1, 0, 1],  # Word 2 Query
        [1, 1, 0, 0]   # Word 3 Query
    ])
    
    # Sample Keys (K)
    K = np.array([
        [1, 0, 1, 0],  # Word 1 Key
        [0, 1, 0, 1],  # Word 2 Key
        [1, 1, 0, 0]   # Word 3 Key
    ])
    
    # Sample Values (V) - Usually carries the content information
    V = np.array([
        [10, 0, 0, 0], # Word 1 Value
        [0, 10, 0, 0], # Word 2 Value
        [0, 0, 10, 0]  # Word 3 Value
    ])
    
    # Add batch dimension: (1, 3, 4)
    Q = np.expand_dims(Q, axis=0)
    K = np.expand_dims(K, axis=0)
    V = np.expand_dims(V, axis=0)
    
    print(f"Shape of Q: {Q.shape}")
    print(f"Shape of K: {K.shape}")
    print(f"Shape of V: {V.shape}")
    
    # --- Run Attention WITHOUT Mask ---
    print("\n--- 1. Running Attention WITHOUT Mask ---")
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\nCalculated Attention Weights:")
    print(np.round(weights, 2))
    # Explanation: 
    # Row 1 (Query 1) should attend most strongly to Key 1 (Column 1) because they are identical [1,0,1,0].
    # Indeed, we expect the diagonal to be highest if Q roughly equals K.
    
    print("\nAttended Output:")
    print(output)
    # Explanation:
    # Since Q1 matches K1, the output for Word 1 is mostly influenced by V1 [10,0,0,0].
    
    # --- Run Attention WITH Mask ---
    print("\n--- 2. Running Attention WITH Mask ---")
    # Let's say we want to mask the 3rd word (e.g., padding token or future token).
    # Mask shape: (1, 3, 3) -> (Batch, Query_Len, Key_Len)
    # 1 means keep, 0 means mask.
    mask = np.array([[
        [1, 1, 0], # Q1 can see K1, K2, but NOT K3
        [1, 1, 0], # Q2 can see K1, K2, but NOT K3
        [1, 1, 0]  # Q3 can see K1, K2, but NOT K3
    ]])
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("\nMasked Attention Weights:")
    print(np.round(weights_masked, 2))
    # Notice the 3rd column is all 0.0 because it was masked out.
    # The probability mass is redistributed to the first two tokens.
    
    print("\nMasked Output:")
    print(output_masked)