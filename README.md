# Continuous Bag of Words and Skipgram word2vec models

This repo contains implementations of CBOW and SkipGram word2vec models, inspired by the original word2vec paper by Mikolov Et al (https://arxiv.org/pdf/1301.3781.pdf). I have also implemented negative sampling and hierarchical softmax as training optimisations.

## Requirements
1. Install Python3.8+ 

## Setup
1. Check out the repo: `git clone git@github.com:PeterPaton123/word2vec.git`
2. Set up a virtual environment for python libraries: `python3 -m venv .`
3. Install those python libraries: `pip install -r requirements.txt`
4. Open the virtual environment: `. venv/bin/activate`

## Testing

The linear correlations in word embeddings can be tested on over 10,000 semantic and syntactic training examples from the original word2vec paper, in `src/semantic_syntactic_tests.py`.
