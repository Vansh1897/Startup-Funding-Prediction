# Startup Funding Prediction

This repository analyzes Indian startup funding records and trains two simple models:

- A binary classifier to estimate the probability of a startup achieving above-median funding (used as a proxy for funding success because the dataset only contains funded startups).
- A regressor to estimate the expected funding amount in USD.

## Getting started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training and evaluation script:
   ```bash
   python model.py
   ```

The script cleans the dataset, engineers year and month features from the deal date, fits models, and prints classification and regression metrics along with the strongest drivers identified by the classifier. It also displays a few sample predictions with success probabilities and expected funding amounts.
