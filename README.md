# Adversarial Machine Learning for Intrusion Detection Systems

## Overview

This research project investigates the robustness of machine-learning-based Intrusion Detection Systems (IDS) against adversarial examples.

The project will first develop conventional ML-based IDS models and evaluate their performance on network traffic. Adversarial examples will then be generated to investigate whether carefully crafted perturbations can cause malicious traffic to be classified as benign.

Finally, adversarial training will be investigated as a defense mechanism, and the robustness of conventional and adversarially trained IDS models will be compared.

## Research Question

How vulnerable are machine-learning-based intrusion detection systems to adversarial examples, and to what extent can adversarial training improve their robustness?

## Research Questions

1. How accurately can conventional ML models detect network intrusions under clean conditions?
2. How does adversarial manipulation affect IDS performance?
3. How does adversarial perturbation strength affect detection performance?
4. Which evaluated IDS model is most vulnerable to the selected adversarial attacks?
5. Can adversarial training improve IDS robustness?
6. What trade-off exists between clean-data performance and adversarial robustness?

## Hypotheses

### H1 — Vulnerability

ML-based IDS models will experience a measurable reduction in detection performance when evaluated against adversarially perturbed network-traffic features.

### H2 — Defense

Adversarial training will improve robustness against adversarial examples compared with conventional training.

### H3 — Trade-off

Improved adversarial robustness may involve some degradation in clean-data performance.

## Dataset

The initial dataset will be CICIDS2017.

The first stage of the project will use binary classification:

* `0` — BENIGN
* `1` — ATTACK

A multiclass formulation may be investigated later.

## Models

The project will investigate:

* Logistic Regression
* Random Forest
* XGBoost
* Neural Network

## Adversarial Attacks

The primary adversarial attacks will be:

* FGSM
* PGD

Additional attacks may be investigated if required by the research results.

## Defense

The primary defense mechanism will be adversarial training.

## Evaluation

Clean and adversarial performance will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* False Positive Rate
* False Negative Rate
* Attack Success Rate

## Project Phases

1. Research Definition & Environment
2. Dataset Collection & Analysis
3. Baseline IDS
4. Baseline Evaluation
5. Adversarial Threat Model
6. Adversarial Example Generation
7. IDS Vulnerability Analysis
8. Adversarial Training
9. Robustness Evaluation
10. Research Analysis and Finalization

## Technology

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* PyTorch
* Adversarial Robustness Toolbox
* Matplotlib
* Jupyter
* Git/GitHub

## Project Structure

```text
adversarial-ids/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── attacks/
│   ├── defenses/
│   ├── evaluation/
│   └── utils/
├── experiments/
├── models/
├── figures/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

## Status

Current phase: **Phase 0 — Research Definition & Environment**
