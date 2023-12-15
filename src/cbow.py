import collections
from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util

def parse_corpus(
    filepath: str,
    occurrence_threshold: int
) -> Dict[str, int]:
    """ Parses each word from a corpus above a given occurrence threshold, for which our model will learn an embedding

    Args:
        filepath (str): The location of the text corpus to learn the word embeddings.
        occurrence_threshold (int): The minimum number of times a word must appear in the corpus to be deemed significant.

    Returns:
        Dict[str, int]: A mapping of significant words to their unique index for one-hot encoding
    """
    word_occurrences = collections.Counter()
    with open(filepath, 'r') as file:
        for line in file:
            word_occurrences.update(line.lower().split())
    vocabulary = {word for word, count in word_occurrences.items() if count >= occurrence_threshold}
    indexed_vocabulary = {word : index for index, word in enumerate(vocabulary)}
    return indexed_vocabulary

def get_batch(
    filepath : str,
    context_size : int,
    vocabulary : Dict[str, int],
    batch_size : int = 128
):
    """
    Generator function which parses each line one at a time as a batch for training.
    Ensures that the context has at least one significant word from the corpus and the target word is significant, too.
    Insignificant context words are indicated by a -1 index, which are disregarded in the model.

    Args:
        filepath (str): The location of the corpus file.
        context_size (int): The number of adjacent words, either side, in the sentence to consider for predicting the current word.
        vocabulary (Dict[str, int]): A dictionary of the one-hot encoding indexes of the significant words.
        batch_size (int): How many training samples we require before yielding a batch.

    Yields:
        Jax.array (batch_size, 2 * context_size): The indexings of the significant word surrounding the training example.
        Jax.array (batch_size): The target word surrounded by the associated indexings.
    """
    with open(filepath, 'r') as file:
        batch_inputs = []
        batch_outputs = []
        for line in file:
            parsed_line = line.lower().split()
            word_indexes = [vocabulary.get(word, -1) for word in parsed_line]
            for i, word_index in enumerate(word_indexes):
                if word_index == -1:
                    continue
                prev_context = word_indexes[max(0, i-context_size):i]
                next_context = word_indexes[i+1:i+context_size+1]
                # Pad prev_context and next_context with -1 if they are shorter than the context size
                prev_context = ([-1] * (context_size - len(prev_context))) + prev_context
                next_context = next_context + ([-1] * (context_size - len(next_context)))
                whole_context = prev_context + next_context
                if (jnp.all(whole_context == -1)):
                    continue
                batch_inputs.append(whole_context)
                batch_outputs.append(word_index)

            while len(batch_inputs) >= batch_size:
                yield batch_inputs[:batch_size], batch_outputs[:batch_size]
                batch_inputs = batch_inputs[batch_size:]
                batch_outputs = batch_outputs[batch_size:]

class CBOW(nn.Module):
    """
    Continuous-bag-of-words model as described in the original word2vec paper (Mikolov Et al., https://arxiv.org/pdf/1301.3781.pdf)
    """
    vocab_size: int
    embedded_dim: int

    def setup(self):
        self.dense = nn.Dense(features=self.vocab_size)

    def __call__(self, x, embeddings):
        # x is a (batch, 2 * context_size) tensor with the context being word indicies in the vocabulary
        
        # Embeddings of the words by indexing the weights matrix (equivalent to matrix multiplication with one-hot encoding)
        valid_indices = x != -1
        int_indices = jnp.where(valid_indices, x, 0)
        selected_embeddings = embeddings[int_indices]
        mask = valid_indices.astype(selected_embeddings.dtype)  
        # Average over the context TODO: try different averaging, perhaps giving closer words greater weightings
        context_vector = jnp.sum(selected_embeddings * mask[:, :, None], axis=1) / jnp.sum(mask, axis=1, keepdims=True) # (batch, embedding_size) 
        # Second linear layer
        logits = self.dense(context_vector) # (batch, vocab_size)
        # Softmax predictions
        return nn.softmax(logits)

def binary_cross_entropy(
    preds : jnp.ndarray, 
    targets : jnp.ndarray
) -> jnp.float64:
    """
    Computes binary-cross entropy loss with the model outputs.

    Args:
        preds (jnp.ndarray): Predicted logits, output of the model.
        targets (jnp.ndarray): Ground truth binary labels.

    Returns:
        Any: The average binary cross-entropy loss for the given predictions and targets.
    """
    epsilon = 1e-12
    preds = jnp.clip(preds, epsilon, 1 - epsilon)
    return -jnp.mean(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))

def train_step(
    params : jnp.ndarray, 
    embeddings : jnp.ndarray, 
    batch_inputs : jnp.ndarray, 
    batch_targets : jnp.ndarray, 
    vocab_size : int, 
    learning_rate : float = 0.1
) -> Tuple[jnp.float64, jnp.ndarray, jnp.ndarray]:
    """
    A single training step over a given batch of training data.

    Args:
        params (jnp.ndarray): Model parameters.
        embeddings (jnp.ndarray): Embedding matrix.
        batch_inputs (jnp.ndarray): Inputs of the current batch.
        batch_targets (jnp.ndarray): Targets of the current batch.
        vocab_size (int): Size of the vocabulary.
        learning_rate (float, optional): Learning rate for the gradient descent. Defaults to 0.1.

    Returns:
        Tuple[jnp.float64, jnp.ndarray, jnp.ndarray]: A tuple containing the loss, updated model parameters,
        and updated embeddings.
    """
    def loss_fn(params, embeddings):
        logits = model.apply({'params': params}, batch_inputs, embeddings)
        one_hot_targets = jax.nn.one_hot(batch_targets, vocab_size)
        return binary_cross_entropy(logits, one_hot_targets)

    gradients_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    (loss, (grads_params, grads_embeddings)) = gradients_fn(params, embeddings)
    new_params = tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads_params)
    new_embeddings = tree_util.tree_map(lambda e, g: e - learning_rate * g, embeddings, grads_embeddings)
    return loss, new_params, new_embeddings

jit_train_step = jax.jit(train_step, static_argnums=4)

def train_step_negative_sampling(
    params : jnp.ndarray, 
    embeddings : jnp.ndarray, 
    batch_inputs : jnp.ndarray, 
    batch_targets : jnp.ndarray, 
    negative_samples : jnp.ndarray, 
    learning_rate : float = 0.1
) -> Tuple[jnp.float64, jnp.ndarray, jnp.ndarray]:
    """
    A single training step over a given batch of training data.

    Args:
        params (jnp.ndarray): Model parameters.
        embeddings (jnp.ndarray): Embedding matrix.
        batch_inputs (jnp.ndarray): Inputs of the current batch.
        batch_targets (jnp.ndarray): Targets of the current batch.
        negative_samples (jnp.ndarray): Chosen negative samples for efficient back propagation.
        learning_rate (float, optional): Learning rate for the gradient descent. Defaults to 0.1.

    Returns:
        Tuple[jnp.float64, jnp.ndarray, jnp.ndarray]: A tuple containing the loss, updated model parameters,
        and updated embeddings.
    """
    def loss_fn(params, embeddings):
        logits = model.apply({'params': params}, batch_inputs, embeddings)
        positive_logits = logits[:, batch_targets]
        negative_logits = logits[:, negative_samples]
        positive_targets = jnp.ones_like(positive_logits)
        negative_targets = jnp.zeros_like(negative_logits)
        return jnp.mean(binary_cross_entropy(positive_logits, positive_targets) + binary_cross_entropy(negative_logits, negative_targets))

    gradients_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    (loss, (grads_params, grads_embeddings)) = gradients_fn(params, embeddings)
    new_params = tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads_params)
    new_embeddings = tree_util.tree_map(lambda e, g: e - learning_rate * g, embeddings, grads_embeddings)
    return loss, new_params, new_embeddings

if __name__ == "__main__":
    # Hyperparameters 
    context_size = 5
    embedded_dim = 100
    num_epochs = 500
    min_frequency = 200

    train_with_negative_sampling = True
    negative_sampling_rng = jax.random.PRNGKey(42)
    num_negative_samples = 20

    filepath = "data/example.txt"
    print("Parsing Corpus")
    vocabulary = parse_corpus(filepath, min_frequency)
    vocab_size = len(vocabulary)

    model = CBOW(vocab_size, embedded_dim)
    embeddings = jax.random.normal(jax.random.PRNGKey(1), (vocab_size, embedded_dim))
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2 * context_size), dtype=jnp.int32), embeddings)['params']
    
    for epoch in range(num_epochs):
        for batch_i, (batch_inputs, batch_targets) in enumerate(get_batch(filepath, context_size, vocabulary)):
            loss_batches = []
            # Perform the training step
            if train_with_negative_sampling:
                # Perform the negative sampling
                negative_sampling_rng, rng = jax.random.split(negative_sampling_rng, 2)
                exclude_positives = jnp.setdiff1d(jnp.arange(vocab_size), jnp.array(batch_targets))
                negative_samples = jax.random.choice(rng, exclude_positives, shape=(num_negative_samples,), replace=False)
                loss, params, embeddings = train_step_negative_sampling(params, embeddings, jnp.array(batch_inputs), jnp.array(batch_targets), negative_samples)
            else:
                loss, params, embeddings = jit_train_step(params, embeddings, jnp.array(batch_inputs), jnp.array(batch_targets), vocab_size)
            loss_batches.append(loss)
        if (epoch % 10 == 0):
            print(f"Epoch {epoch} average batch loss: {jnp.mean(jnp.array(loss_batches))}")


    # Example testing the quality of our learned embeddings on various semantic and syntatic linear relationships

    from semantic_syntactic_tests import test_accuracy
    
    test_accuracy("data/question-words.txt", embeddings, vocabulary)
    