# Heterogeneous Graph Neural Networks with Weak Information
## Abatract
Heterogeneous Graph Neural Networks (HGNNs) have achieved significant success in various heterogeneous graph learning tasks.
Traditional HGNNs assume that the input heterogeneous graph data is complete or sufficient. The performance of HGNNs will be affected when the input heterogeneous graph data suffers from weak information, such as incomplete structure, incomplete features, and insufficient labels.
In this paper, we propose the Heterogeneous Graph Neural Network with Weak Information (HGNN-WI) to address multiple types of weak information for data simultaneously. Specifically, we utilize a graph contrastive learning strategy within a self-supervised framework to achieve structural fusion and feature completion. We tackle the problem of weak structure by extracting subgraphs using metapaths of the heterogeneous graph. The subgraphs are fused to enhance the graph's structural integrity. For the weak feature of the graph data, the k-Nearest Neighbors (kNN) algorithm is used to generate an augmented graph that facilitates feature propagation. HGNN-WI employs subgraph-based and augmented graph-based feature complementation to enrich node features. For the weak labels problem, we use a contrastive learning strategy with group discrimination to produce robust node embeddings.
Experimental results on DBLP, ACM, and IMDB datasets demonstrate that HGNN-WI outperforms eight baseline algorithms, confirming its effectiveness in handling weak information in heterogeneous graphs.

## Requirements

- Python==3.9.0
- Pytorch==1.12.0
- Networkx==2.8.4
- numpy==1.22.3
- dgl==0.9.0
- scikit-learn==1.1.1
- scipy==1.7.3

## Running the code
python main.py
