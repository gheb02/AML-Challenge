# AML-Challenge

## Submission
--------------------------

This repository contains the code and final submission notebooks for the DATABROS participation in the Advanced Machine Learning and Compuer Vision challenge.

## Repository Structure
-------------------------

That sounds like a great project! Here is a suggested structure and content for your group repository's README, tailored for an advanced Machine Learning challenge/hackathon.

🤖 Group Name - [Challenge Name] Submission

A repository containing the source code and final submission notebook for our team's participation in the [Challenge Name] advanced machine learning challenge/hackathon.

🏆 Final Result and Code

This section highlights the most important files related to our performance and final solution.
File Name	Description	Purpose
Final Notebook.ipynb	Our best submission.	The final, documented solution used for the ultimate leaderboard ranking. It consolidates the best-performing model, preprocessing, and submission generation.
Second Submission.ipynb	Our second-best submission.	A notebook detailing the runner-up approach, which performed well on the public leaderboard. Useful for comparing different strategies.

💻 Repository Structure

Our codebase is organized into modular files to handle data preparation, modeling, and utility functions efficiently.

    utils.py: A comprehensive utility script containing various functions for:

        Data Preparation: Loading, cleaning, and feature engineering.

        Submission Generation: Functions to format predictions into the required submission file.

        Evaluation: Metrics and routines for cross-validation and final result evaluation.

        Grid Search: Implementation of the grid search logic used for hyperparameter optimization.

    model.py: The core of our machine learning solution, comprising:

        Best Model Architectures: Definitions of the winning model structure(s) (e.g., custom CNN, Transformer variant, LightGBM configuration).

        Loss Functions: Implementations of the specific loss functions used for training.

        Training Functions: Reusable routines for training, validation, and managing the model lifecycle.
