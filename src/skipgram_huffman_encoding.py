import collections
import heapq
from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util

def parse_corpus(
    filepath: str,
    occurrence_threshold: int
) -> Tuple(Dict[int, int], Dict[str, int]):
    """ Parses each word from a corpus above a given occurrence threshold, for which our model will learn an embedding

    Args:
        filepath (str): The location of the text corpus to learn the word embeddings.
        occurrence_threshold (int): The minimum number of times a word must appear in the corpus to be deemed significant.

    Returns:
        Dict[str, int]: A mapping of significant words to their unique index for one-hot encoding.
        Dict[str, int]: A mapping of significant words to their occurrence count for use in Huffman tree.
    """
    word_occurrences = collections.Counter()
    with open(filepath, 'r') as file:
        for line in file:
            word_occurrences.update(line.lower().split())
    filtered_occurrences = {word : count for word, count in word_occurrences.items() if count >= occurrence_threshold}
    vocabulary = {word for word, _count in filtered_occurrences.items()}
    indexed_vocabulary = {word : index for index, word in enumerate(vocabulary)}
    filtered_occurrences = {index : count for index, (_word, count) in enumerate(filtered_occurrences.items())}
    return filtered_occurrences, indexed_vocabulary

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
                yield jnp.array(batch_inputs[:batch_size], dtype=jnp.int32), jnp.array(batch_outputs[:batch_size], dtype=jnp.int32)
                batch_inputs = batch_inputs[batch_size:]
                batch_outputs = batch_outputs[batch_size:]

def build_huffman_tree(
    word_freqs : Dict[int, int]
) -> Dict[int, str]:
    """
    Generates huffman encodings of signfiifcant words in the corpus using priority queues.
    Args:
        word_freqs (Dict[int, int]): A dictionary mapping word indices to their frequencie in the corpus.

    Returns:
        Dict[int, str]: A dictionary mapping word indices to their Huffman encodings.
    """
    # Create a priority queue from word frequencies
    heap = [[weight, [word, ""]] for word, weight in word_freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # The final element of heap contains the root of the Huffman tree
    huffman_tree = sorted(heap[0][1:], key=lambda p: (len(p[1]), p))

    # Create a dictionary to map words to their Huffman codes
    huffman_code = {word: code for word, code in huffman_tree}
    return huffman_code

def get_inner_node_indexings(
    huffman_tree : Dict[int, str]
) -> Dict[str, int]:
    """
    Traverses a given dictionary of leaf nodes of a huffman encodings and finds all the internal nodes. Enumerates them.

    Args:
        huffman_tree (Dict[int, str]): A mapping of word indexes to their respective huffman encodings.

    Returns:
        Dict[str, int]: A mapping of the internal node encoding to their own enumeration (indexing).
    """
    inner_nodes = {""} # The root node with no decision
    for index, leaf_code in huffman_tree.items():
        curr_sub_encoding = leaf_code[:-1]
        while (curr_sub_encoding not in inner_nodes and curr_sub_encoding != ""):
            inner_nodes.add(curr_sub_encoding)
            curr_sub_encoding = curr_sub_encoding[:-1]
    return {inner_code : index for index, inner_code in enumerate(inner_nodes)}

def forward(
    target_embeddings : jnp.ndarray, 
    context_embeddings : jnp.ndarray, 
    batch_targets : jnp.ndarray, 
    batch_contexts : jnp.ndarray, 
    index_to_huffman_code : Dict[int, str], 
    inner_node_indexing : Dict[str, int]
) -> float:
    """
    Performs a forward pass of the SkipGram model using Huffman Encodings. For each target word and associated context, finds the target embedding vector and traverses the Huffman-encoding tree calculating traversal probabilities on the way to each leaf word in the context.
    
    Args:
        target_embeddings (jnp.ndarray): Learned embeddings of target words.
        context_embeddings (jnp.ndarray): Learned embeddings of context words.
        batch_targets (jnp.ndarray): Indices of target words in the current batch.
        batch_contexts (jnp.ndarray): Indices of the associated context words in the current batch. An indices of -1 indicates an invalid or infrequent context.
        index_to_huffman_code (Dict[int, str]): A dictionary mapping word indices to their Huffman encodings.
        inner_node_indexing (Dict[str, int]): A dictionary mapping partial Huffman code sequences to indices of internal nodes in the Huffman tree.

    Returns:
        float: The sum of the negative log probabilities for all the target-context pairs in the batch. This value serves as a loss measure for the SkipGram model using Huffman Encoding, where lower values indicate better model performance.
    """
    log_traversal_probs = []
    for (target, context) in zip(batch_targets, batch_contexts):
        # Embeddings of the words by indexing the weights matrix (equivalent to matrix multiplication with one-hot encoding)
        target_embedding = target_embeddings[target]
        for context_index in context:
            context_index = context_index.item()
            if context_index == -1:
                continue
            context_huffman_code = index_to_huffman_code[context_index]
            log_prob = jnp.array(0.0)
            # Traverse the Huffman key
            for i in range(len(context_huffman_code)):
                # Retrieve the embedding of the internal node
                node_index = inner_node_indexings[context_huffman_code[:i]]
                node_embedding = context_embeddings[node_index]
                # Calculate the probability contribution of this node
                direction = 1 if context_huffman_code[i] == '1' else -1
                log_prob += -jnp.log(jax.nn.sigmoid(direction * jnp.dot(target_embedding, node_embedding)))
        log_traversal_probs.append(log_prob)
    return jnp.sum(jnp.array(log_traversal_probs))

def train_step(
    target_embeddings, 
    context_embeddings, 
    batch_targets, 
    batch_context, 
    index_to_huffman_code, 
    inner_node_indexing, 
    learning_rate = 0.1
):
    def loss_func(target_embeddings, context_embeddings):
        return forward(target_embeddings, context_embeddings, batch_targets, batch_contexts, index_to_huffman_code, inner_node_indexing)

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
    filtered_occurrences, indexed_vocabulary = parse_corpus(filepath, min_frequency)
    print(filtered_occurrences)
    vocab_size = len(indexed_vocabulary) 
    index_to_huffman_code = build_huffman_tree(filtered_occurrences)
    print(index_to_huffman_code)
    inner_node_indexings = get_inner_node_indexings(index_to_huffman_code)
    # Binary tree invariant, there are always num_leaf_node - 1 inner nodes
    print(f"Number of huffman codes: {len(index_to_huffman_code)}, Number of inner nodes: {len(inner_node_indexings)}")
    
    target_embedding_init_rng, context_embedding_init_rng, rng = jax.random.split(rng, 3)
    target_embeddings = jax.random.normal(target_embedding_init_rng, (vocab_size, embedding_dim))
    context_embeddings = jax.random.normal(context_embedding_init_rng, (vocab_size - 1, embedding_dim))
    
    for epoch in range(num_epochs):
        batch_losses = []
        for batch_targets, batch_contexts in get_batch(filepath, context_size, indexed_vocabulary, batch_size=10):            
            loss, target_embeddings, context_embedding_ = train_step(target_embeddings, context_embeddings, batch_targets, batch_contexts, index_to_huffman_code, inner_node_indexings)
            batch_losses.append(loss)
        if (epoch % 1 == 0):
            print(f"Epoch: {epoch + 1}, Loss: {jnp.mean(jnp.array(batch_losses))}")
    