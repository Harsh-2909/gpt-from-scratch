# GPT from Scratch

## Overview
GPT from Scratch is a project aimed at implementing the GPT (Generative Pre-trained Transformer) model from scratch. The project focuses on understanding the inner workings of the GPT model and providing a step-by-step guide for its implementation.

<!-- This project follows the [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) architecture, which is a scaled-down version of the original GPT model. The GPT-2 model consists of a stack of transformer blocks, each containing a multi-head self-attention mechanism and a feed-forward neural network. -->

This project follows the [Transformer](https://arxiv.org/pdf/1706.03762) architecture, which is a deep learning model that uses self-attention mechanisms to capture long-range dependencies in sequential data. The transformer model has been widely used in natural language processing tasks, such as machine translation and text generation.

> Note: This project follows Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) video, which provides a detailed explanation of the GPT model and its implementation.

## Setup
To set up the project, follow these steps:

1. Clone the repository:
    ```shell
    git clone https://github.com/username/gpt-from-scratch.git
    ```

2. Create a virtual environment using venv:
    ```shell
    python3 -m venv gpt-env
    ```

3. Activate the virtual environment:
    ```shell
    source gpt-env/bin/activate
    ```

4. Install the required dependencies:
    ```shell
    pip install -r requirements.txt
    ```

## Running the Project
To run the project, execute the following command:

```shell
python main.py
```

This will train the GPT model on the provided training data and generate text using the trained model.

The parameters for the GPT model can be modified in the `main.py` file, such as the number of layers, hidden size, and number of attention heads.
To run it in a device without a good gpu just for testing purpose, you can change the hyperparameters in the `main.py` file to a smaller value.

```python
# Hyper Parameters
batch_size: int = 32  # How many sequences to process in parallel
# Number of tokens processed at a time. Content length of predictions
block_size: int = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32  # Number of embeddings
n_head = 4  # Number of heads in the multi-head attention
n_layer = 6
dropout = 0.2  # Drops out 20% of the neurons every forward and backward pass
```

## Code Structure
The project is structured as follows:

```
gpt-from-scratch/
│
├── data/
│   ├── data.txt
│
├── src/
|   ├── gpt.ipynb
│   ├── main.py
│
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
```

The `data` directory contains the training data for the GPT model. The `src` directory contains the source code for the GPT model implementation. The `main.py` file contains the GPT model implementation while `gpt.ipynb` contains the notes and approach to implement the GPT model. It also has notes on why the model is implemented in a certain way.

## Resources
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing
Contributions are welcome! Please refer to the [contribution guidelines](CONTRIBUTING.md) for more information.
