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
    """
    Parses each word from a corpus above a given occurrence threshold, for which our model will learn an embedding

    Args:
        filepath (str): The location of the text corpus to learn the word embeddings.
        occurrence_threshold (int): The minimum number of times a word must appear in the corpus to be deemed significant.

    Returns:
        Dict[str, int]: A mapping of significant words to their unique index for one-hot encoding index.
    """
    word_occurrences = collections.Counter()
    with open(filepath, 'r') as file:
        for line in file:
            word_occurrences.update(line.lower().split())
    filtered_occurrences = {word : count for word, count in word_occurrences.items() if count >= occurrence_threshold}
    indexed_vocabulary = {word : index for index, (word, _count) in enumerate(filtered_occurrences.items())}
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
                batch_inputs.append(word_index)
                batch_outputs.append(whole_context)

            while len(batch_inputs) >= batch_size:
                yield jnp.array(batch_inputs[:batch_size]), jnp.array(batch_outputs[:batch_size])
                batch_inputs = batch_inputs[batch_size:]
                batch_outputs = batch_outputs[batch_size:]

class SkipGram():
    def forward(self, 
        target_embeddings : jnp.ndarray,
        context_embeddings : jnp.ndarray,
        batch_targets : jnp.ndarray, 
        batch_context : jnp.ndarray
    ):
        """
        Forward pass for the SkipGram model.

        Args:
            target_embeddings (jnp.ndarray): Embeddings for target words with shape (vocab_size, embedding_dim).
            context_embeddings (jnp.ndarray): Embeddings for context words with shape (vocab_size, embedding_dim).
            batch_targets (jnp.ndarray): Indices of target words in the batch, shape (batch_size,).
            batch_context (jnp.ndarray): Indices of context words for each target word in the batch, shape (batch_size, 2 * context_window).
        Returns:
            jnp.ndarray: Sigmoid activations for target-context pairs, shape (batch_size, 2 * context_window).
        """
        batch_target_embeddings = target_embeddings[batch_targets]
        batch_context_embeddings = context_embeddings[batch_context]
        alignments = jnp.matmul(batch_context_embeddings, batch_target_embeddings[:,:,jnp.newaxis]).squeeze()
        return jax.nn.sigmoid(alignments)

    def train_step(self, 
        target_embeddings : jnp.ndarray, 
        context_embeddings : jnp.ndarray, 
        batch_targets : jnp.ndarray, 
        batch_context : jnp.ndarray, 
        negative_samples : jnp.ndarray, 
        learning_rate : float = 0.1
    ):
        """
        Performs a single training step for the SkipGram model, using positive samples from the context window and randomly generated negative samples.

        Args:
            target_embeddings (jnp.ndarray): The embeddings for target words, shape (vocab_size, embedding_dim).
            context_embeddings (jnp.ndarray): The embeddings for context words, shape (vocab_size, embedding_dim).
            batch_targets (jnp.ndarray): Indices of target words in the batch, shape (batch_size,).
            batch_context (jnp.ndarray): Indices of context words for each target word in the batch, shape (batch_size, context_window), with -1 indicating padding or infrequent context words.
            negative_samples (jnp.ndarray): Indices of negative sample words for each target word, shape (batch_size, context_window).
            learning_rate (float, optional): The learning rate for the gradient descent update. Defaults to 0.1.

        Returns:
            float: The computed loss for the current training step.
            jnp.ndarray: Updated embeddings for target words, shape (vocab_size, embedding_dim).
            jnp.ndarray: Updated embeddings for context words, shape (vocab_size, embedding_dim).
        """
        def loss_func(target_embeddings, context_embeddings):
            positive_probabilities = self.forward(target_embeddings, context_embeddings, batch_targets, batch_context)
            negative_probabilities = self.forward(target_embeddings, context_embeddings, batch_targets, negative_samples)
            mask = jnp.where(batch_context == -1, 0, 1)
            positive_loss = -mask * jnp.log(positive_probabilities)
            negative_loss = -mask * jnp.log(1 - negative_probabilities)
            return jnp.mean(positive_loss + negative_loss)

        loss, gradients = jax.value_and_grad(loss_func, (0, 1))(target_embeddings, context_embeddings)
        target_embeddings -= learning_rate * gradients[0]
        context_embeddings -= learning_rate * gradients[1]
        return loss, target_embeddings, context_embeddings

if __name__ == "__main__":
    # Hyperparameters
    context_size = 2
    embedding_dim = 10
    num_epochs = 30
    min_frequency = 2
    
    rng = jax.random.PRNGKey(42)
    rng, model_init_rng = jax.random.split(rng, 2)

    filepath = "data/example.txt"
    indexed_vocabulary = parse_corpus(filepath, min_frequency)
    vocab_size = len(indexed_vocabulary)

    model = SkipGram()
    target_embedding_init_rng, context_embedding_init_rng, rng = jax.random.split(rng, 3)
    target_embeddings = jax.random.normal(target_embedding_init_rng, (vocab_size, embedding_dim))
    context_embeddings = jax.random.normal(context_embedding_init_rng, (vocab_size, embedding_dim))

    for epoch in range(num_epochs):
        batch_losses = []
        for batch_i, (batch_targets, batch_context) in enumerate(get_batch(filepath, context_size, indexed_vocabulary, batch_size=10)):
            rng, negative_sample_rng = jax.random.split(rng, 2)  
            unique_context = jnp.unique(batch_context)
            unique_context = unique_context[unique_context != -1]
            not_in_context = jnp.arange(vocab_size)[~jnp.isin(jnp.arange(vocab_size), unique_context)]
            negative_samples = jax.random.choice(negative_sample_rng, not_in_context, shape = jnp.shape(batch_context), replace=True)
            loss, target_embeddings, context_embeddings = model.train_step(target_embeddings, context_embeddings, batch_targets, batch_context, negative_samples)
            batch_losses.append(loss)
        if (epoch % 1 == 0):
            print(f"Epoch: {epoch + 1}, Loss: {jnp.mean(jnp.array(batch_losses))}")