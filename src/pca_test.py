import jax.numpy as jnp

from typing import Tuple

def pca(
    embeddings : jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Performs principle component analysis (PCA) on the 

    Args:
        embeddings : (jnp.ndarray) The learned embeddings from training a word2vec model. (vocab_size, embedding_dim)
    
    Returns:
        jnp.ndarray : The sorted principle eigenvalues of the covariance matrix of embeddings, in descending order. The proportion of the total sum represents the component's significance.
        jnp.ndarray : The corresponding basis components for the sorted eigenvalues.
    """
    mean = jnp.mean(embeddings, axis=0)
    std_dev = jnp.std(embeddings, axis=0)
    standardized_embeddings = (embeddings - mean) / std_dev
    covariance_matrix = jnp.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and principle components in descending order (order of contribution to total variance)
    sorted_indices = jnp.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors

def log_embedding_quality(
    principle_eigenvalues : jnp.ndarray
) -> None:
    """
    To assess how evenly distributed the eigenvalues from PCA are, we examine what proportion of the total variance is accounted for by a certain proportion of the largest eigenvalues. 
    For evenly significant components these will be approximately equal.

    Args:
        principle_eigenvalues (jnp.ndarray) : The eigenvalues corresponding to the principle components, from PCA.

    Returns:
        None: This function does not return any values, it logs test results to the output, however.
    """
    total_variance = jnp.sum(eigenvalues)
    num_components = len(principle_eigenvalues)

    # Compute the cumulative variance explained by the eigenvalues
    cumulative_variance = jnp.cumsum(eigenvalues) / total_variance

    # Proportions of variance explained
    variance_explained = {
        "50%": (jnp.searchsorted(cumulative_variance, 0.5) + 1)/num_components,
        "70%": (jnp.searchsorted(cumulative_variance, 0.7) + 1)/num_components,
        "90%": (jnp.searchsorted(cumulative_variance, 0.9) + 1)/num_components,
        "95%": (jnp.searchsorted(cumulative_variance, 0.95) + 1)/num_components
    }

    for (variance_proportion, eigenvalue_proportion) in variance_explained:
        print(f"{variance_proportion} of the variance is comprised of {eigenvalue_proportion} of the eigenvalues (proportion)")
