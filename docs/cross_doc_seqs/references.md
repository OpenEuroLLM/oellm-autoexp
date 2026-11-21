# References

Every paper listed here was checked on arXiv on 2026-11-21. The claims quoted are from its abstract
unless a section is named.

## Packing related documents together

**In-Context Pretraining: Language Modeling Beyond Document Boundaries.** W. Shi, S. Min, M. Lomeli,
C. Zhou, M. Li, G. Szilvasy, R. James, X. V. Lin, N. A. Smith, L. Zettlemoyer, S. Yih, M. Lewis. ICLR
2024, [arXiv:2310.10638](https://arxiv.org/abs/2310.10638).
- **Method:** reorders the pretraining corpus so a context holds related documents. It finds
  approximate nearest neighbours (dense retrieval embeddings), then makes a graph traversal that
  visits every document once, so no document is repeated.
- **Result:** LLaMA-architecture models up to 7B, compared with random packing: in-context learning
  +8 %, reading comprehension +15 %, faithfulness to previous contexts +16 %, long-context reasoning
  +5 %, retrieval augmentation +9 %.
- **Here:** the template for step 4. Our traversal is hierarchical (coarse → fine clusters → greedy
  chains over centroids), because a window has tens of millions of documents. They deduplicate before
  packing; we defer copies and near duplicates instead (see [adaptation.md](adaptation.md)).

**Analysing the Impact of Sequence Composition on Language Model Pre-Training.** Y. Zhao, Y. Qu,
K. Staniszewski, S. Tworkowski, W. Liu, P. Miłoś, Y. Wu, P. Minervini. ACL 2024,
[arXiv:2402.13991](https://arxiv.org/abs/2402.13991).
- **Findings:**
  - plain causal masking over a packed sequence of unrelated documents adds distracting context and
    hurts downstream performance;
  - intra-document masking removes that;
  - packing related documents reduces the noise.
- **Method:** BM25Chunk, a retrieval-based sequence construction. It reports in-context learning
  +11.6 %, knowledge memorisation +9.8 %, context utilisation +7.2 %.
- **Here:** the main argument that **removing the mask is only safe together with relatedness**. It
  is also the reason for the `group` boundary mode.

**Structured Packing in LLM Training Improves Long Context Utilization (SPLiCe).** K. Staniszewski,
S. Tworkowski, S. Jaszczur, Y. Zhao, H. Michalewski, Ł. Kuciński, P. Miłoś. AAAI 2025,
[arXiv:2312.17296](https://arxiv.org/abs/2312.17296).
- **Method:** builds training sequences from retrieved related documents.
- **Result:** at 3B, 7B and 13B, better context use on long-context tasks (Qasper, HotpotQA), less
  lost-in-the-middle. Even brief fine-tuning with SPLiCe is enough, and structured code data also
  helps natural-language tasks.
- **Here:** a short phase of structured packing on top of a pretrained model already pays. Our
  setting, a late switch inside a long run, is close to theirs.

## Masking, truncation, sequence composition

**How to Train Long-Context Language Models (Effectively) (ProLong).** T. Gao, A. Wettig, H. Yen,
D. Chen. [arXiv:2410.02660](https://arxiv.org/abs/2410.02660).
- **Finding:** §6.1, "disabling cross-document attention in continued training benefits both the
  short and long-context performance", also faster.
- **Here:** the **strongest evidence against** dropping the mask. Their packing is random (unrelated
  documents). It is the reason the switch here is gated on a measured comparison, not assumed
  ([adaptation.md](adaptation.md), checks).

**The Llama 3 Herd of Models.** A. Grattafiori, A. Dubey, et al.
[arXiv:2407.21783](https://arxiv.org/abs/2407.21783).
- **What it says:** Llama 3 masks attention between documents of a sequence. The pretraining section
  reports limited impact in standard pre-training, but important for continued pretraining on very
  long sequences. This is from the body, not the abstract.
- **Here:** at 64k with mostly short documents, unmasked random packing is the regime where the mask
  matters most.

**Fewer Truncations Improve Language Modeling (Best-fit Packing).** H. Ding, Z. Wang, G. Paolini,
V. Kumar, A. Deoras, D. Roth, S. Soatto. ICML 2024, [arXiv:2404.10830](https://arxiv.org/abs/2404.10830).
- **Finding:** concatenate-and-chunk truncates many documents. Best-fit packing avoids it at the same
  efficiency: reading comprehension +4.7 %, context following +16.8 %, program synthesis +9.2 %,
  closed-domain hallucination down by up to 58.3 %.
- **Here:** our cut is concatenate-and-chunk, like Megatron's, so truncation is unchanged. We do
  merge the pieces of a document that the window reads in two slots. Best-fit would need padding
  (no token is free to fill a gap) or variable-length sequences. It is a possible later refinement.

**Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum.** H. Pouransari,
C.-L. Li, J.-H. R. Chang, P. K. A. Vasu, C. Koc, V. Shankar, O. Tuzel. NeurIPS 2024,
[arXiv:2405.13226](https://arxiv.org/abs/2405.13226).
- **Method:** buckets documents by length and trains with variable sequence lengths.
- **Result:** an 8k-context 1B model at the cost of a 2k baseline, up to 6× faster to target accuracy.
- **Here:** the alternative answer to the same straggler problem: make the sequences cheap instead of
  filling the idle time. Not compatible with the fixed-shape 64k step of the main run without a
  larger change.

## Code and repository context

**DeepSeek-Coder: When the Large Language Model Meets Programming.** D. Guo, Q. Zhu, D. Yang, et al.
[arXiv:2401.14196](https://arxiv.org/abs/2401.14196).
- **What it does:** the abstract says it trains on a project-level code corpus. The data section
  (body) builds repository-level samples, with the files of a repository ordered by their
  dependencies.
- **Here:** the model for a metadata group key in code (repo), see [metadata.md](metadata.md).

## Attention behaviour

**Efficient Streaming Language Models with Attention Sinks.** G. Xiao, Y. Tian, B. Chen, S. Han,
M. Lewis. ICLR 2024, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
- **Finding:** initial tokens receive disproportionate attention ("sinks") whatever their content.
- **Here:** with document masking, every document's first token (the BOS) was its sink. Without the
  mask, later documents see earlier BOS tokens and the sequence's first token. See
  [adaptation.md](adaptation.md), failure case F3.

## Duplicates and embeddings

**SemDeDup: Data-efficient learning at web-scale through semantic deduplication.** A. Abbas,
K. Tirumala, D. Simig, S. Ganguli, A. S. Morcos. [arXiv:2303.09540](https://arxiv.org/abs/2303.09540).
- **Finding:** embedding-space near duplicates can be removed with little loss.
- **Here:** our near-duplicate test uses the same signal (cosine in an embedding space). We defer,
  never drop: the window's data stays exactly the same.

**Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.** Y. Zhang, M. Li,
D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou.
[arXiv:2506.05176](https://arxiv.org/abs/2506.05176).
- **What it is:** 0.6B / 4B / 8B embedding models, strong on multilingual MTEB and code retrieval,
  Apache 2.0.
- **Here:** the 0.6B model is the default embedder: it covers the mix's 20 % multilingual share and
  its code.
