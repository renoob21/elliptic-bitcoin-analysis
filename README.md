# Elliptic Dataset Graph-Based Neural Networks

This is my attempt on Elliptic Bitcoin Dataset fraud prediction.

Written in [Rust](https://rust-lang.org/), utilizing [Burn](https://burn.dev/)'s neural-network utilities with WGPU backend.

## Current Approach Done
### Simple Graph Neural Network

Using 2 separate `Linear` layers to:
1. Process `Node`'s own feature
2. Process `Node` neighbors' aggregated features

Then the two layers is aggregated and passed to the final output layer.

Current development's result metric (at 10 epochs):

| Metric         | Min.     | Epoch    | Max.     | Epoch    |
|----------------------------------|----------|----------|----------|----------|
| **Train**      |||||
| F1 Score       | 72.328   | 1        | 90.417   | 10       |
| Precision      | 69.632   | 1        | 86.782   | 10       |
| Recall         | 86.957   | 1        | 96.678   | 10       |
| **Validation** |||||
| F1 Score       | 51.762   | 1        | 73.141   | 10       |
| Precision      | 56.703   | 1        | 70.271   | 10       |
| Recall         | 75.876   | 1        | 82.642   | 10       |


### Concatenated Neighbor's Aggregated Features

Instead of directly aggregating all features, neighbor's features are concatenated into main `Node`'s features. Expanding its 2nd dimension (from 165 to 330) before being passed to the neural network. It is done to prevent the `Node`'s features being dilluted by its neighbors', which might result in the actual `Node` losing its own identity. This approach is inspired by [GraphSAGE](https://arxiv.org/abs/1706.02216)'s take on preprocessing graph features.

Current development's result metric (at 10 epochs):

| Metric                           | Min.     | Epoch    | Max.     | Epoch    |
|----------------------------------|----------|----------|----------|----------|
| **Train** ||||||
| F1 Score       | 75.414   | 1        | 92.499   | 10       |
| Precision      | 72.120   | 1        | 89.604   | 10       |
| Recall         | 88.002   | 1        | 97.141   | 10       |
| **Valid** ||||||
| F1 Score       | 58.407   | 1        | 75.084   | 10       |
| Precision      | 59.092   | 1        | 73.471   | 10       |
| Recall         | 79.615   | 1        | 82.788   | 4        |
