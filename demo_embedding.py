import sys
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load the tokenizer and model
#model_name = "Mistral-7B-v0.1"  # Or the specific Mistral model you're using
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example text
text = "This is an example sentence."

prompt_template = "Q: when account identity is {}, educational is {}, incoming is {}, credit is {}, shall we approve the credit card application? A: {}"
text = prompt_template.format("1001", "High School", "1000000", "600", "Y")

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt")  # pt for PyTorch tensors

# Calculate the cosine similarity between the first two token embeddings
with torch.no_grad():
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    print(token_embeddings.shape)
    token_embeddings_mean = token_embeddings.mean(dim=1)
    print(token_embeddings_mean.shape)

    first_token_embedding = token_embeddings[0][0]
    second_token_embedding = token_embeddings[0][1]

    # Convert the PyTorch tensors to numpy arrays
    first_token_embedding = first_token_embedding.numpy()
    second_token_embedding = second_token_embedding.numpy()

    # Calculate the cosine similarity
    similarity = cosine_similarity(first_token_embedding, second_token_embedding)
    print(similarity)

# Get the model's output (including embeddings)
with torch.no_grad(): # ensures that gradients are not calculated or stored, saving memory and speeding up the computation
    outputs = model(**inputs)

# Access the token embeddings
token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

# If you only have one sentence (batch size 1), you can simplify:
token_embeddings = token_embeddings.squeeze(0)  # Remove the batch dimension

# token_embeddings now contains the embeddings for each token in the input text.
# To see the shape:
print(token_embeddings.shape)  # Example: torch.Size([6, 4096]) (6 tokens, 4096-dimensional embeddings)

# To access the embedding of a specific token (e.g., the first token):
first_token_embedding = token_embeddings[0]
print(first_token_embedding.shape) # Example: torch.Size([4096])

# If you want the *word* embeddings rather than the subword embeddings, you might need to do some averaging or pooling.
# For example, if "programming" is split into "program" and "##ming", you'd combine their embeddings.

# Example of getting the embedding matrix (the word embeddings):
embedding_matrix = model.get_input_embeddings().weight

print(embedding_matrix.shape) # Example: torch.Size([32000, 4096]) (32000 words in vocab, 4096-dimensional embeddings)
