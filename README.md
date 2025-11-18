# AML-Challenge

This repository contains the code and final submission notebooks for the DATABROS participation in the Advanced Machine Learning and Compuer Vision challenge.

- **Gabriele Cabibbo 2196717**
- **Emanuele Gallo 2197051**
- **Giorgio Taramanni 1961217**

## Final Model Architectures

We implemented a bottleneck architecture composed of an encoder that projects the input embeddings. This architecture first increases the dimensionality from 1024 to 1408 in a first linear layer, and then decreases it back from 1408 to 1024 to create the bottleneck and moderate overfitting. The decoder comprises two layers that sequentially increase the dimensionality of the representation from 1024 to 1408, until the output layer projects it to the final output dimension 1536. Each linear layer, made exception for the output layer, is followed by a Layer Normalization and a GELU activation function, and a dropout probability of 0.35 is applied after each nonlinearity.

The eight best-performing models were then ensembled by making the final prediction using a weighted average of the normalized outputs of each single model. Specifically, weights were assigned to each model proportional to the Mean Reciprocal Rank (MRR) obtained by that model during validation. This is the model which obtained the best results in the public leaderboard.

## Repository Structure

The repository contains:

    utils.py: A comprehensive utility script containing various functions for:

        Data Preparation

        Submission Generation

        Evaluation

        Grid Search for Hyperparameters Selection

    model.py: The core of our solution, comprising:

        Best Model Architectures

        Loss Functions

        Training Functions
        
## Project Report

A detailed explanation of the project's methodology, experiments, results, and conclusions can be found in the [Final Project Report](<Final Report>).
