# LLM Notes
## LLM
Large language models are AI systems that are designed to process and analyze vast amounts of natural language data and then use that information to generate responses to user prompts.
## LLMOps
Large Language Model Ops (LLMOps) encompasses the practices, techniques and tools used for the operational management of large language models in production environments.
## RAG (Retrieval Augmented Generation)
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. 
(Using a pretrained model and adding external data to it to make it more specifically useful to the user)

## Evaluating Text Generation
-  **BLEU: Bilingual Evaluation Understudy** provides a score to compare sentences  The idea behind BLEU is to count the matching n-grams in the sentences. A unigram is a word (token), a bigram is a pair of two words and so on. The order of the words is irrelevant in this case.
-   **Cosine Similarity Score** measure of how similar the ideas and concepts represented in two pieces of text are.

## Rouge Score
-   f1-score = 2 * (Precision*Recall)/(Precision+Recall)

-   precision = TP/(TP+FP)

-   Recall = TP/(TP+FN)

The following five evaluation metrics are available.

- ROUGE-N: Overlap of n-grams[2] between the system and reference summaries.
- ROUGE-1 refers to the overlap of unigrams (each word) between the system and reference summaries.
- ROUGE-2 refers to the overlap of bigrams between the system and reference summaries.
- ROUGE-L: Longest Common Subsequence (LCS)[3] based statistics. Longest common subsequence problem takes into account sentence-level structure similarity naturally and identifies longest co-occurring in sequence n-grams automatically.
- ROUGE-W: Weighted LCS-based statistics that favors consecutive LCSes.
- ROUGE-S: Skip-bigram[3] based co-occurrence statistics. Skip-bigram is any pair of words in their sentence order.
- ROUGE-SU: Skip-bigram plus unigram-based co-occurrence statistics.