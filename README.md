# Anonymous-Walk-Embeddings-for-Graph-Level-Prediction

Implemented a machine learning pipeline based on Anonymous Walk Embeddings (AWE) for graph-level binary classification on the OGB-MolHIV dataset. Converted molecular graphs into structural embedding vectors using random anonymous walks and trained a feedforward neural network optimized with ROC-AUC for imbalanced learning.

---

## Project Overview

This project builds a structural graph representation using Anonymous Walk Embeddings (AWE) and applies supervised learning for molecular property prediction.

Task:

- Binary classification  
- Predict whether a molecule inhibits HIV  

Dataset:

- OGB-MolHIV (Open Graph Benchmark)  
- Graph-level prediction task  
- Severe class imbalance  

The system avoids message-passing GNNs and instead uses structural pattern frequencies extracted from anonymous walks.

---

## Dataset

Dataset used: OGB – ogbg-molhiv  

- Molecular graphs  
- Nodes represent atoms  
- Edges represent chemical bonds  
- Binary label: HIV inhibitor (1) or not (0)  

Splits:

- Train  
- Validation  
- Test  

Predefined splits from OGB are used to ensure standardized evaluation.

---

## System Pipeline

The complete pipeline consists of:

1. Dataset Loading & Split Preparation  
2. Random Walk Generation  
3. Anonymous Walk Conversion  
4. Vocabulary Construction  
5. Graph-to-Embedding Conversion  
6. Feedforward Neural Network Classifier  
7. Training with Class Imbalance Handling  
8. Evaluation using ROC-AUC and PR Metrics  

---

## 1. Anonymous Walk Representation

### Anonymous Walk Concept

A node walk:

```
[5, 9, 5, 3]
```

Is converted into an anonymous walk:

```
[0, 1, 0, 2]
```

This removes node identity and keeps only structural pattern information.

Anonymous walks capture structural signatures independent of specific node labels.

---

## 2. Random Walk Generation

Random walks are generated using:

- Edge index adjacency list  
- Random starting node  
- Uniform neighbor sampling  
- Fixed walk length  

Parameters used for embedding:

```
walk_length = 10
num_walks = 300
```

---

## 3. Anonymous Walk Vocabulary

A vocabulary of unique anonymous walk patterns is constructed from training graphs.

Steps:

- Generate random walks  
- Convert to anonymous form  
- Store unique patterns  
- Assign each pattern an index  

This creates a structural dictionary used to represent graphs.

---

## 4. Graph-to-Embedding Conversion

Each graph is converted into a fixed-length vector:

- Count frequency of anonymous walk patterns  
- Normalize counts into probability distribution  

Result:

```
Graph → AWE Vector (size = vocabulary size)
```

This produces a dense structural embedding.

---

## 5. AWEClassifier Neural Network

A feedforward neural network is used since AWE produces fixed-length embeddings.

### Architecture

Input: Anonymous Walk Embedding  

Hidden Layers:

- Linear( input_dim → 512 )  
- ReLU  
- BatchNorm  
- Dropout(0.4)  

- Linear(512 → 256)  
- ReLU  
- BatchNorm  
- Dropout(0.3)  

- Linear(256 → 64)  
- ReLU  

Output:

- Linear(64 → 1)  

No message passing is required since structural encoding is precomputed.

---

## 6. Handling Class Imbalance

MolHIV dataset is heavily imbalanced.

Solution:

- Compute positive class weight  
- Use `BCEWithLogitsLoss` with `pos_weight`  

This ensures the model gives higher importance to minority positive samples.

---

## 7. Training Strategy

Optimizer:

```
Adam (lr = 3e-4)
```

Learning Rate Scheduler:

```
ReduceLROnPlateau
```

Early Stopping:

- Patience = 5  
- Save best model based on validation ROC-AUC  

Training Epochs:

```
Up to 20 epochs
```

Evaluation metric:

```
ROC-AUC
```

Chosen due to strong class imbalance.

---

## 8. Evaluation Metrics

The model is evaluated using:

- ROC-AUC  
- Precision-Recall Curve  
- Average Precision (AP)  
- Confusion Matrix  
- Training Loss Curves  

Precision-Recall metrics are emphasized since ROC can be misleading in imbalanced datasets.

---

## Results & Observations

- Training and validation curves show stable convergence  
- No severe overfitting observed  
- Validation ROC-AUC improves steadily before early stopping  
- Test ROC-AUC confirms generalization  

Precision-Recall analysis indicates meaningful discriminative ability despite imbalance.

---

## Visualization Outputs

The project includes plots for:

- Training Loss vs Epochs  
- Validation ROC-AUC vs Epochs  
- ROC Curve (Test Set)  
- Precision-Recall Curve  
- Confusion Matrix  
- Prediction Probability Distribution  

These visualizations provide deeper insight into model behavior.

---

## Key Insights

- Anonymous Walk Embeddings effectively capture structural graph patterns  
- Structural frequency representations can compete without deep message passing  
- Class imbalance handling is critical for molecular datasets  
- ROC-AUC is appropriate for imbalanced graph classification  
- Early stopping improves generalization  

---

## Conclusion

This project demonstrates a complete machine learning pipeline for graph-level prediction using Anonymous Walk Embeddings.

Instead of relying on GNN message passing, the model leverages structural walk patterns to create expressive fixed-length representations.

The approach:

- Captures meaningful structural signals  
- Handles class imbalance effectively  
- Achieves stable convergence  
- Demonstrates strong generalization on OGB-MolHIV  

This work highlights the power of structural graph representations for molecular activity prediction.
