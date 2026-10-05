use std::collections::HashMap;

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use petgraph::graph::{DiGraph, NodeIndex};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[derive(Debug, Clone)]
pub struct EllipticItem {
    pub tx_id: u64,
    pub time_step: usize,
    pub features: [f32; 165],
    pub class: u32,
}

pub type BitcoinGraph = DiGraph<EllipticItem, ()>;

#[derive(Debug, Clone)]
pub struct GraphData {
    pub graph: BitcoinGraph,
    // id_map: HashMap<u64, NodeIndex>,
    train_idxs: Vec<NodeIndex>,
    test_idxs: Vec<NodeIndex>,
}

impl GraphData {
    pub fn load(split_idx: usize) -> Self {
        let mut x_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path("data/elliptic_txs_features.csv")
            .unwrap();
        let mut y_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path("data/elliptic_txs_classes.csv")
            .unwrap();

        let mut graph = BitcoinGraph::new();
        let mut id_map = HashMap::new();
        let mut train_idxs = Vec::new();
        let mut test_idxs = Vec::new();

        for (x_res, y_res) in x_reader.records().zip(y_reader.records()) {
            let x_recs = x_res.unwrap();
            let mut x_recs = x_recs.iter();
            let tx_id = x_recs.next().map(|x| x.parse::<u64>().unwrap()).unwrap();
            let time_step = x_recs.next().map(|x| x.parse::<usize>().unwrap()).unwrap();
            let features = x_recs
                .map(|x| x.parse::<f32>().unwrap())
                .collect::<Vec<f32>>();
            let features = features.try_into().unwrap();
            let y_recs = y_res.unwrap();
            let class_str = y_recs.iter().last().unwrap();
            let class = match class_str {
                "1" => Some(0),
                "2" => Some(1),
                _ => None,
            };

            let node_idx = graph.add_node(EllipticItem {
                tx_id,
                time_step,
                features,
                class: class.unwrap_or(0),
            });

            id_map.insert(tx_id, node_idx);
            if class.is_some() {
                if time_step <= split_idx {
                    train_idxs.push(node_idx);
                } else {
                    test_idxs.push(node_idx);
                }
            }
        }

        let mut edge_reader = csv::ReaderBuilder::new()
            .from_path("data/elliptic_txs_edgelist.csv")
            .unwrap();

        for edge in edge_reader.records() {
            let edge = edge.unwrap();
            let mut edge = edge.iter();
            let a = id_map[&edge.next().unwrap().parse().unwrap()];
            let b = id_map[&edge.next().unwrap().parse().unwrap()];
            graph.add_edge(a, b, ());
        }

        Self {
            graph,
            // id_map,
            train_idxs,
            test_idxs,
        }
    }

    pub fn train(&self) -> Vec<NodeIndex> {
        self.train_idxs.clone()
    }

    pub fn test(&self) -> Vec<NodeIndex> {
        self.test_idxs.clone()
    }
}

#[derive(Debug, Clone)]
pub struct GraphBatch<B: Backend> {
    pub node_features: Tensor<B, 2>,
    pub neighbor_features: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, NodeIndex, GraphBatch<B>> for GraphData {
    fn batch(&self, node_indices: Vec<NodeIndex>, device: &Device<B>) -> GraphBatch<B> {
        let batch_size = node_indices.len();

        let result: Vec<_> = node_indices
            .par_iter()
            .map(|&idx| {
                let current_node = &self.graph[idx];

                let mut neighbor_sum = [0.0; 165];
                let mut count = 0.0;

                for neighbor_idx in self.graph.neighbors(idx) {
                    let neighbor_data = &self.graph[neighbor_idx];
                    for i in 0..165 {
                        neighbor_sum[i] += neighbor_data.features[i];
                    }
                    count += 1.0;
                }

                if count > 0.0 {
                    for i in 0..165 {
                        neighbor_sum[i] /= count;
                    }
                }

                (current_node.features, neighbor_sum, current_node.class)
            })
            .collect();

        let mut flat_nodes = Vec::with_capacity(batch_size * 165);
        let mut flat_neighbors = Vec::with_capacity(batch_size * 165);
        let mut targets = Vec::with_capacity(batch_size);

        for (features, neighbors, target) in result {
            flat_nodes.extend_from_slice(&features);
            flat_neighbors.extend_from_slice(&neighbors);
            targets.push(target);
        }

        let node_features =
            Tensor::from_floats(TensorData::new(flat_nodes, vec![batch_size, 165]), device);
        let neighbor_features = Tensor::from_floats(
            TensorData::new(flat_neighbors, vec![batch_size, 165]),
            device,
        );
        let targets = Tensor::from_ints(TensorData::new(targets, vec![batch_size]), device);

        GraphBatch {
            node_features,
            neighbor_features,
            targets,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data::GraphData;

    #[test]
    fn test_data_loading() {
        GraphData::load(34);
    }
}
