# DeepSeek-v4: mHC + Engram Memory Architecture

### **Architecture Overview**
This model integrates the core innovations from DeepSeek-v4 into a 16MB parameter-efficient GPT baseline:

1.  **Engram Memory Architecture:** 
    *   Adds a "second axis of sparsity" via a **512k-entry hash table**.
    *   Uses a **bigram hash** of the current and previous token to index directly into a repository of static knowledge.
    *   This offloads factual retrieval from the Transformer blocks, allowing them to focus entirely on reasoning.

2.  **Manifold-Constrained Hyper-Connections (mHC):**
    *   Replaces the standard residual stream with **3 parallel streams** (Attention, MLP, and Identity/Memory).
    *   Includes a **Birkhoff Polytope Mixer** at the end of every block.
    *   Uses the **Sinkhorn-Knopp algorithm** during the forward pass to constrain the mixing matrix to be doubly stochastic, ensuring training stability even with high-dimensional parallel flows.

### **Technical Specs**
*   **Tokenizer:** SP8192 (8,192 vocab size).
*   **Hidden State:** 512-dim.
*   **Streams:** 3 (mHC-stabilized).
*   **Memory:** 512k Engram Table (4-dim embeddings).
*   **Budget:** 16MB (Int6/Int8 Quantized).

### **Training Strategy**
*   **Optimizer:** Muon (matrices) + Adam (scalars/embeddings/engram).
*   **Batch Size:** 2M tokens (optimized for 8xH100).
*   **Schedule:** 20k steps with 0.35 recurrence start fraction.
