# Drafted: Additional Benchmark Questions

> **Verification Status**: Verified against `papers/extracted_papers.json` text.

### Question 8: Label Smoothing Effect
- **Question**: What is the label smoothing value ($\epsilon_{ls}$) used during Transformer training, and what effect did the authors observe on perplexity and BLEU score?
- **Expected Answer**: The Transformer uses label smoothing of value $\epsilon_{ls} = 0.1$. The authors noted that this hurts perplexity as the model learns to be more unsure, but improves accuracy and BLEU score [1706.03762v7, Page 8].
- **Key Facts**: ["0.1", "perplexity", "accuracy", "bleu"]
- **Expected Page**: 1706.03762v7, Page 8
- **Type**: answerable

### Question 9: Maximum Path Length Comparison
- **Question**: In Table 1 of "Attention Is All You Need", what is the maximum path length between any two positions for a Self-Attention layer compared to a Recurrent layer?
- **Expected Answer**: Self-Attention has a maximum path length of $O(1)$, whereas a Recurrent layer has a maximum path length of $O(n)$ [1706.03762v7, Page 6].
- **Key Facts**: ["o(1)", "o(n)"]
- **Expected Page**: 1706.03762v7, Page 6
- **Type**: answerable

### Question 10: Positional Encoding Formula
- **Question**: What exact mathematical formula is used for sinusoidal positional encodings at even dimension indices ($2i$) in Attention Is All You Need?
- **Expected Answer**: The formula is $PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})$ [1706.03762v7, Page 6].
- **Key Facts**: ["sin", "10000", "dmodel"]
- **Expected Page**: 1706.03762v7, Page 6
- **Type**: answerable

### Question 11: Helium and Moshi Parameter Counts
- **Question**: What are the parameter counts of the Helium language model and the Moshi speech-text model in the Moshi paper?
- **Expected Answer**: The Helium language model has 7 billion parameters (7B), and Moshi is a 7B-parameter speech-text foundation model (specifically 7.4B total) [2410.00037v1, Page 1, Page 8].
- **Key Facts**: ["7b", "7 billion"]
- **Expected Page**: 2410.00037v1, Page 1 or 8
- **Type**: answerable

### Question 12: Dual-Stream Audio Modeling in Moshi
- **Question**: How does Moshi model parallel audio streams, and whose speech channels do they represent?
- **Expected Answer**: Moshi models audio streams in parallel representing both the user audio (input) and Moshi's audio (output), enabling full-duplex conversational interaction [2410.00037v1, Page 1, Page 3, Page 4].
- **Key Facts**: ["user", "moshi", "stream", "duplex"]
- **Expected Page**: 2410.00037v1, Page 1, 3, or 4
- **Type**: answerable

### Question 13: Full-Duplex Fine-Tuning Dataset
- **Question**: What conversational dataset was used to fine-tune Moshi to gain its full-duplex spoken dialogue capabilities?
- **Expected Answer**: Moshi was fine-tuned on the Fisher dataset (Cieri et al., 2004) to obtain its full-duplex capabilities [2410.00037v1, Page 8].
- **Key Facts**: ["fisher"]
- **Expected Page**: 2410.00037v1, Page 8
- **Type**: answerable

### Question 14: Dynamic Chunking Compression Resolution
- **Question**: What compression ratio (in bytes per chunk) does Dynamic Chunking naturally learn, and what standard tokenization format does it match?
- **Expected Answer**: Dynamic Chunking naturally compresses data to 4.5–5 bytes/chunk, matching the resolution of Byte-Pair Encoding (BPE) tokenizers [2507.07955v2, Page 3].
- **Key Facts**: ["4.5-5", "bpe", "tokenizer"]
- **Expected Page**: 2507.07955v2, Page 3
- **Type**: answerable

### Question 15: Smoothing Module EMA Formula
- **Question**: What formula defines the Exponential Moving Average (EMA) applied by the smoothing module in Dynamic Chunking?
- **Expected Answer**: The smoothing module defines EMA as $\bar{z}_t = P_t \hat{z}_t + (1 - P_t)\bar{z}_{t-1}$ [2507.07955v2, Page 7, Equation 5].
- **Key Facts**: ["p_t", "pt", "ema", "(5)", "equation 5"]
- **Expected Page**: 2507.07955v2, Page 7
- **Type**: answerable

### Question 16: Head Count Comparison Across Papers
- **Question**: Compare the attention head count of the Transformer base model with the head count in Moshi's Depth Transformer.
- **Expected Answer**: The Transformer base model has 8 attention heads [1706.03762v7, Page 9], while Moshi's Depth Transformer uses 16 attention heads [2410.00037v1, Page 8].
- **Key Facts**: ["8", "16"]
- **Expected Page**: 1706.03762v7, Page 9; 2410.00037v1, Page 8
- **Type**: multi-paper

### Question 17: ResNet Bottleneck Architecture Trap
- **Question**: What expansion ratio is used in the bottleneck residual blocks of the ResNet-50 backbone cited in Attention Is All You Need?
- **Expected Answer**: Refuse; the papers in the library do not contain information on ResNet-50 bottleneck block expansion ratios.
- **Key Facts**: []
- **Expected Page**: None
- **Type**: unanswerable

### Question 18: Moshi Training Cost Trap
- **Question**: What was the total financial training cost in USD or total megawatt-hours of power consumed to train the Moshi 7B foundation model?
- **Expected Answer**: Refuse; the Moshi paper reports hardware and training duration (e.g. GPU days), but does not report dollar cost or total megawatt-hours consumed.
- **Key Facts**: []
- **Expected Page**: None
- **Type**: unanswerable

### Question 19: English-to-French Table 3 Ablation Near-Miss
- **Question**: In Table 3 of Attention Is All You Need, what English-to-French BLEU score is reported for the Transformer architecture variation with 16 attention heads (row A)?
- **Expected Answer**: Refuse; Table 3 of Attention Is All You Need only reports metrics on the English-to-German development set (newstest2013), and does not report English-to-French BLEU scores for the architecture variations in row A.
- **Key Facts**: []
- **Expected Page**: None
- **Type**: unanswerable, near-miss

### Question 20: Translation Trap in Dynamic Chunking
- **Question**: What BLEU score does Dynamic Chunking reach on the English-to-Japanese translation benchmark?
- **Expected Answer**: Refuse; the Dynamic Chunking paper does not report English-to-Japanese machine translation benchmarks.
- **Key Facts**: []
- **Expected Page**: None
- **Type**: unanswerable
