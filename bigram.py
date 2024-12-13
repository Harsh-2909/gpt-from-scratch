import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)  # Manually seeding to get deterministic random numbers

# Get the data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r') as f:
    text = f.read()

# Getting the vocabulary of the training data
char = sorted(list(set(text)), key=None)
vocab_size = len(char)

# Next we create a tokenizer to tokenize the inputs
# A tokenizer simply converts the inputs into a list of integers where each integer represent a token (which can be a word, sub-word, or character)
# There are multiple types of tokenizers like word, sub-word, and character based tokenizer.
# Here, we will be using a character tokenizer.

stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}
def encode(s): return [stoi[ch] for ch in s]
def decode(token_list): return ''.join([itos[i] for i in token_list])


# Next, we encode the entire text and convert it into tensors using PyTorch
encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long)
# print(data.shape, data.dtype)
# data[:1000]

# split the data into train and validation chunks
train_data_percent = 0.9  # 90% would be for training
n = int(train_data_percent * len(data))
train_data = data[:n]
val_data = data[n:]

# We cannot train a model with all the training data at once, as it is very expensive computationally. Thus, we train the model chunk by chunk where each chunk is taken out randomly from the data. Here, the chunk size will be indicated by the variable `block_size`.

#   The training chunk of block size gives example to the model on how to respond when facing a set of inputs. For example, here when [18] is sent as input then return 47.
#   Similarly, when [18, 47] is sent as input, then return 56. Thus, this gives us `block_size` number of examples from context length of 1 to block_size.

#   Here, we will also need to add batch_size to do parallel processing to make training more efficient. Each block in a batch is processed parallelly without any interference.

torch.manual_seed(1337)
batch_size = 4  # The number of parallel independent sequences
block_size = 8  # The number of tokens processed at a time. Maximum context length of predictions


def get_batch(split_type):
    """Generates a small batch of data of inputs x and target y"""
    data = train_data if split_type == "train" else val_data
    # generates a list of offsets randomly of size `batch_size`
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    # Here, we are creating the context blocks
    x = torch.stack([data[i: i+block_size] for i in ix])
    # Target Blocks. This should be the output when a context block is passed to the model
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch("train")


class BigramLanguageModel(nn.Module):
    """We are using the Bigram language model. It is a part of n-gram models. A bigram model uses the context of the previous 1 token to predict the next token. It does not check the context beyond 1 token."""

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # Returns a tensor of dim (batch_size, block_size, vocab_size) (B, T, C). Here its, [4, 8, 65]
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Converting 3D array to 2D array where B and T dimensions are converted to a single dimension while preserving the C dimension.
            # This is done to conform with the cross_entropy documentation.
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # loss should be -ln(1/vocab_size). Check Karpathy videos for more understanding.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """idx is (B, T) array. This function will take the input tokens `idx` and generate the output tokens based on the inputs.

        Arguments:
            idx {Array} -- An array of size (B, T)
            max_new_tokens {Integer} -- The number of tokens to generate

        Returns:
            [Array] -- The output of size (B, T+max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # First get the prediction using the forward function.
            logits, loss = self(idx)
            # Then only keep the last index of the T dimension because each of the element in T represents a token and we are only focused om the last token for each prediction.
            logits = logits[:, -1, :]  # Shape: (B, C)
            # Apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # Shape: (B, C)
            # Get the sample for the distribution.
            # After the softmax, we are getting the a 2D array where each row element (which denotes the batch) has an array of length C.
            # That array is the probability distribution of what would be the next token for each batch.
            # Now, the multinomial function samples the probability distribution and returns the single token from the probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)
            # Append the prediction to the input sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
# logits, loss = model(xb)
# print(logits.shape)
# print(loss)

# Generates token 0 which corresponds to newline. This will be used to start generation
idx = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = model.generate(idx, max_new_tokens=100)[0]
output = decode(generated_tokens.tolist())
print(output)
