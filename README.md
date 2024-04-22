# RNN Explainability

This project focuses on studying and implementing various explainability techniques for Recurrent Neural Networks (RNNs). RNNs are a type of artificial neural network designed to recognize patterns in sequences of data, such as text. While they are incredibly powerful, they can also be difficult to interpret, which is where this project comes in.

This project mainly focuses on RNNs used for natural language processing (NLP) tasks, such as text classification or sentiment analysis. By understanding how RNNs make predictions and implementing various explainability techniques, we can gain insights into how these models work and why they make certain predictions.


## Project Objectives

The main objectives of this project are:

1. To understand how RNNs make predictions, which can be crucial for trust in AI systems.
2. To implement various RNN explainability techniques and compare their effectiveness.
3. To provide a resource for others who want to learn about RNN explainability and implement these techniques in their own projects.

## Techniques Studied

We separated the explainability techniques into two categories: model-specific and model-agnostic. Model-specific techniques are designed to work with specific types of models, such as RNNs, while model-agnostic techniques can be applied to any type of model.

### Model-Agnostic Techniques
- Permutation importance
- Likelihood importance
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)

### Model-Specific Techniques
- Integrated Gradients
- Smooth Gradients
- Sailency Maps

## Getting Started

To get started with this project, you'll need create an environment and install the required dependencies. You can do this by running the following commands:

```bash
# Clone the repository
git clone https://github.com/AsterisCrack/rnn_explainability

# Navigate to the project directory
cd rnn_explainability

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

Once you've installed the dependencies, you need to download the dataset used in this project. You can do this by running the following command:

```bash
python -m src.RNNModelTrain.download_extract_data
```

This will download the dataset needed and the pre-trained embeddings used for our RNN model.

Now, we recommend training the RNN model first before running the explainability techniques. You can also use your own RNN model if you prefer. To train the RNN model, run the following command:

```bash
# Train the normal RNN model
python -m src.RNNModelTrain.train

# Train the detached embeddings RNN model
# This is used for the model specific techniques
python -m src.RNNModelDetachedEmbeddings.train
```

Once you've trained the RNN model, you can start exploring all the explainability techniques implemented in this project. You can run each technique separately by running the notebooks located in the `notebooks` `AgnosticModels` and `SpecificModels` directories.

## Contributing

If you'd like to contribute to this project, feel free to fork this repository and submit a pull request. You can also open an issue if you have any suggestions or ideas for improvement.

## License

This project is licensed under the MIT License
