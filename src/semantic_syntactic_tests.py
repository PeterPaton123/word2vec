import collections
from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util

def cosine_simularity(
    vec_1 : jnp.ndarray,
    vec_2 : jnp.ndarray
) -> jnp.float64:
    return jnp.dot(vec_1 / jnp.linalg.norm(vec_1), vec_2 / jnp.linalg.norm(vec_2))

def test_accuracy(
    test_filepath : str,
    learned_embeddings : jnp.ndarray,
    vocabulary : Dict[str, int]
) -> None:
    """
    Tests the accuracy of learned embeddings by evaluating linear relationships 
    such as 'King' - 'Man' + 'Woman' = 'Queen'. It logs the mean, standard deviation, 
    and various percentiles of the cosine similarity correlations across the test set.

    Args:
        test_filepath (str) : The file path for the test data, with each line containing a tuple of words to test the relationship.
        learned_embeddings (jnp.ndarray): The array of learned word embeddings.
        vocabulary (Dict[str, int]): A dictionary mapping words to their indices in the learned embeddings array.

    Returns:
        None: This function does not return any value, it logs test statistics to the output, however.
    """
    correlations = []
    with open(test_filepath, 'r') as file:
        for example in file:
            word_indexes = [vocabulary.get(word, -1) for word in example.lower().split()]
            # Skip examples with words not in our vocabulary
            if (jnp.any(jnp.array(word_indexes) == -1)): 
                continue
            correlation = cosine_simularity(learned_embeddings[word_indexes[0]] - learned_embeddings[word_indexes[1]] + learned_embeddings[word_indexes[2]], learned_embeddings[word_indexes[3]])
            correlations.append(correlation)

    percentiles = [10, 25, 50, 75, 90]
    percentile_values = jnp.percentile(input, percentiles)
    print(f"Test Correlations Output\nMean: {jnp.mean(correlations)}\nStandard Deviation: {jnp.std(correlations)}")
    for percentile, value in zip(percentiles, percentile_values):
        print(f"{percentile}th Percentile: {value}")

