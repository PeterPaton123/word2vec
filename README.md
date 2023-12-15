# Continuous Bag of Words and Skipgram word2vec models

This repo contains implementations of CBOW and SkipGram word2vec models, inspired by the original word2vec paper by Mikolov Et al (https://arxiv.org/pdf/1301.3781.pdf). I have also implemented negative sampling and hierarchical softmax as training optimisations.

## Requirements
1. Install Python3.8+ 

## Setup
1. Check out the repo: `git clone git@github.com:PeterPaton123/word2vec.git`
2. Set up a virtual environment for python libraries: `python3 -m venv .`
3. Install those python libraries: `pip install -r requirements.txt`
4. Open the virtual environment: `. venv/bin/activate`

## Models

### CBOW

In a naive implementation, continuous bag of words model takes a context as an input and returns an output over the whole probability distribution. We can optimise this training by only selecting positive targets and a subset of negative samples for which to predict over. This is negative sampling and is implemented in `src/cbow.py`.

### Skipgram

The skipgram model has two implementations. A implementation using negative sampling, which takes each target word and surrounding context and finds their associated target and context embeddings. The sigmoid of the dot product of each of these vectors returns the probability of the context word appearing in the context window of the surrounding word. Negative samples are also fed in as a separate context, so the model learns to discrimniate between unrelated words. In a naive implementation of SkipGram, a probability distribution over the whole vocabulary is returned for each word in the context window. As the vocabulary is often very large (30,000 in the original Skipgram implementation), this is highly inefficient; hierarchical softmax aims to approximate this vector by encoding the vocabulary into a binary tree (a Hoffman encoding) and uses each target word to predict a series of traversals through the tree.

## Testing

The linear correlations in word embeddings can be tested on over 10,000 semantic and syntactic training examples from the original word2vec paper, in `src/semantic_syntactic_tests.py`.
