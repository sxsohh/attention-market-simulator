# Attention Market Simulator

**Author:** Stefan Soh

## Overview

This project models human attention as a market microstructure system. The core idea is to treat attention like a tradable asset, where boredom and fatigue act as drift forces, notifications and high-impact posts behave like aggressive orders, and the platform algorithm acts as an adaptive market maker optimizing for long-term engagement.

The simulation generates synthetic behavioral data with different regimes (engaged, fatigued, overstimulated, addictive loop, disengaged, baseline), which I then use to build a complete machine learning research pipeline including:

- Feature engineering inspired by quantitative finance
- Regime classification using Random Forest
- Short-horizon attention forecasting
- 5-step forward return regression
- Reinforcement learning agent that learns notification policies
- Statistical analysis and visualizations

This project demonstrates a full quant research workflow similar to what you would build at a trading firm or for a quantitative research internship.

## Core Concepts

### Attention as a Market
The model treats attention dynamics like asset price movements:
- **Attention level** corresponds to mid-price
- **Boredom/fatigue** act as drift and decay forces
- **Dopamine hits** create volatility shocks
- **Imbalance** is calculated as (demand - liquidity) / (demand + liquidity)
- **Regimes** represent different behavioral states, analogous to market regimes

### The Algorithm as a Market Maker
The platform algorithm adjusts its behavior adaptively:
- Increases notification frequency when users are disengaged
- Reduces aggressiveness when users are fatigued or overstimulated
- Optimizes for long-term engagement while avoiding user burnout

### Emergent Market Phenomena
The simulation exhibits realistic patterns found in financial markets:
- Volatility clustering
- Regime shifts and transitions
- Imbalance as a predictor of future attention changes
- Heavy-tailed return distributions

## Features

The feature engineering pipeline creates signals similar to those used in quantitative trading:
- 1-step and 5-step attention returns
- Forward returns for prediction tasks
- Rolling volatility, mean, and absolute returns
- Imbalance metrics and changes
- Liquidity and demand ratios
- Lagged features (1, 2, 3, 5, 10 timesteps)
- Regime labels for classification

## Machine Learning Models

### Regime Classification
Random Forest classifier that predicts behavioral regime based on microstructure features.
- **Classes:** engaged, fatigued, overstimulated, addictive_loop, disengaged, baseline
- **Evaluation:** accuracy, confusion matrix, classification report

### Attention Direction Classifier
Binary classifier predicting whether attention will increase or decrease in the next timestep.
- **Model:** Random Forest
- **Metrics:** accuracy, F1 score, precision, recall

### 5-Step Forward Regression
Regression model predicting the magnitude of future attention changes.
- **Model:** Random Forest Regressor
- **Metrics:** R², MAE, RMSE
- **Visualizations:** predicted vs actual plots, residual analysis

## Reinforcement Learning

I implemented a Q-learning agent that learns to control the algorithm's aggressiveness through trial and error:
- **Actions:** decrease / maintain / increase aggressiveness
- **State space:** discretized attention and boredom levels (6 states)
- **Reward function:** balanced to encourage attention while penalizing fatigue and boredom

The agent trains over 200 episodes and learns a policy that balances short-term engagement with long-term user retention.

## Usage

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the simulation
```bash
python run_simulation.py
```
This generates the dataset in `data/attention_simulation.csv` and creates basic plots.

### Generate visualizations
```bash
python generate_visuals.py
```
Saves all plots to the `visuals/` directory.

### Train the RL agent
```bash
python train_rl_agent.py
```
Trains a Q-learning agent and saves the Q-table to `models/`.

### Jupyter notebooks
The `notebooks/` directory contains detailed analysis:
1. `01_explore_simulation.ipynb` - Data exploration and visualization
2. `02_feature_engineering.ipynb` - Feature creation and correlation analysis
3. `03_models.ipynb` - Model training, evaluation, and comparison

## Project Structure

```
attention-market-simulator/
├── src/
│   ├── environment.py      # Main simulation environment
│   └── rl_env.py           # RL wrapper for the environment
├── notebooks/              # Jupyter notebooks for analysis
├── data/                   # Generated simulation data
├── visuals/                # Output plots and figures
├── models/                 # Saved models (Q-table, etc.)
├── run_simulation.py       # Main simulation script
├── generate_visuals.py     # Visualization generation
├── train_rl_agent.py       # RL training script
└── requirements.txt        # Python dependencies
```

## Results

The simulation produces realistic attention dynamics with clear regime transitions. Model performance varies by run due to the stochastic nature of the simulation, but typical results include:
- **Regime classifier:** 70-80% accuracy
- **Regression model:** R² around 0.6-0.7
- **RL agent:** successfully learns to balance engagement and fatigue avoidance

Visualizations include:
- Time series of attention, volatility, imbalance, boredom, and fatigue
- Regime-colored scatter plots
- Return distributions showing fat tails
- Correlation heatmaps of features
- Imbalance vs future attention change scatter plots

## Technical Details

### Dependencies
- **numpy, pandas** - data manipulation
- **matplotlib, seaborn** - visualization
- **scikit-learn** - machine learning models
- **jupyter** - interactive notebooks

Full dependency list available in `requirements.txt`.

### Simulation Parameters
The simulation uses market-inspired parameters:
- Boredom growth rate: 0.05 per timestep
- Fatigue growth rate: 0.02 per timestep
- Dopamine decay: 0.1
- Volatility decay: 0.05
- Adaptive algorithm with dynamic notification bias

## Future Extensions

Potential improvements and extensions:
- Implement deep RL algorithms (DQN, PPO, A3C)
- Add social network effects and user interactions
- Model multiple heterogeneous users
- Incorporate more realistic event distributions
- Add hyperparameter optimization
- Extend to multi-platform scenarios
- Include temporal patterns (time of day, day of week effects)

## Notes

This project is a research prototype designed to explore the intersection of market microstructure theory and behavioral modeling. The goal is to demonstrate quantitative research skills and the ability to translate concepts from one domain (finance) to another (user behavior).

The code prioritizes clarity and educational value. While the models show promising patterns, they would require significant refinement and validation for production use.
