### Introduction

Quantization refers to the process of reducing the precision of the weights, activations, and biases of a neural network by reducing the number of bits used to store the network parameters. It reduces the resources required to serve a model and is vital to the deployment of neural networks in resource constrained environments.

This project aims to compare a basic neural network trained on the MNIST dataset with the  TensorFlow Lite (TFLITE) and quantized TFLITE forms of the same network. Additionally, the project explores model compression by pruning a neural network, and comparing the compressibility of the pruned network to that of the original one.

Pruning was achieved by removing connections (weights) based on their importance or contribution to the model's overall performance. 

### Usage

To use this project, 

- First clone the repository:

```bash
git@github.com:Aldion0731/neural-net-quantization-and-pruning.git
```

- Second, install requirements

```bash
pipenv sync
```

- Finally, navigate to `src/notebooks/quantization_and_pruning.ipynb` and run the code in the notebook.

### Results

#### Table 1 - Model Comparison

![Results_1](/results/model-comparison.png)

- The table above summarizes the sizes and accuracies for the various model formats generated during the project.

- From the table, we can clearly see that quantization can result in a 4x reduction in without significant reduction in accuracy.

- We can also see that pruning may lead to improved may lead to improved accuracy due to potential regularization effects.

- Despite not seeing differences in the post training quantized model and the quaantized aware trained model, available research indicates that there will be improved accuracies for quantized aware trainied models. This case be due to the limited training process.

#### Table 2 - Model Compression Table

![Results_1](/results/zipped-comparison.png)

- The table above shows the difference in compressibility between the baseline model and the pruned baseline model. This is due to the prevalence of sparse tensors in the pruned model and the relative ease in compressing sparse tensors.


### Further Improvements

- This project serves as a starting point for exploring model compression techniques on the MNIST dataset. The accuracy of the baseline model was not prioritized. As a result of this, the baseline model was trained for a single iteration of gradient descent. Further improvements could include:

    - Training model for more epochs
    - Trying different pruning strategies
    - Trying different pruning schedules, pruning ratios and sparsity targets.