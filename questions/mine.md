# Mine: Benchmark Questions

> **Origin Note**: These 7 questions were drafted for me by Claude (an AI assistant) and I am verifying them against the ground-truth extracted papers.
> **Verification Status**: UNVERIFIED until Step D2.

1. What optimizer settings (beta1, beta2, epsilon) and warmup steps did the Transformer use for training? Expected: Adam, beta1=0.9, beta2=0.98, epsilon=1e-9, 4000 warmup steps (Attention, section 5.3). Type: answerable.
2. For the base Transformer, how many layers, what model dimension, how many attention heads, and what feed-forward size? Expected: N=6, d_model=512, h=8, d_ff=2048 (Attention, section 3 and Table 3). Type: answerable, multi-fact.
3. What frame rate and bitrate does Moshi's Mimi codec use, and what latency does Moshi report? Expected: about 12.5 Hz, 1.1 kbps, roughly 160 ms theoretical and 200 ms practical latency (Moshi). Type: answerable, verify.
4. In Dynamic Chunking (version 2), what two components decide where chunk boundaries fall? Expected: routing module decides boundaries (via cosine similarity between adjacent vectors); the downsampler selects boundary vectors; smoothing module is in dechunking layer (Dynamic Chunking, section 2.2, pages 3, 6-7). Type: answerable, version-sensitive.
5. What top-1 accuracy does the Transformer reach on ImageNet? Expected: refuse; not in the papers. Type: unanswerable.
6. What BLEU score does the Transformer get on WMT 2014 English-to-Czech? Expected: refuse; the paper reports only EN-DE and EN-FR. Type: unanswerable, near miss.
7. How well does Moshi handle Hindi conversations? Expected: refuse; verify the Moshi paper has no Hindi results. Type: unanswerable.
