"""
Graph Encoder for HOLOGRAPH.

Encodes causal graph structure into embeddings for:
- Graph-level representation
- Node-level features with structural context
- Edge-level features for causal relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import centralized constants (P1: Remove magic numbers)
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    EDGE_THRESHOLD,
    FEATURE_EXTRACTION_THRESHOLD,
)

from .causal_state import CausalState


@dataclass
class GraphEncoderConfig:
    """Configuration for graph encoder."""
    node_dim: int = 128  # Node embedding dimension
    edge_dim: int = 64  # Edge feature dimension
    hidden_dim: int = 256  # Hidden layer dimension
    num_layers: int = 3  # Number of GNN layers
    dropout: float = 0.1
    use_edge_features: bool = True
    aggregation: str = "mean"  # mean, sum, max


class GraphConvLayer(nn.Module):
    """
    Graph convolution layer for causal graphs.

    Handles both directed edges (W) and bidirected edges (M).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 64,
        use_edge_features: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_features = use_edge_features

        # Node transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # Message transformation for directed edges
        self.directed_msg = nn.Sequential(
            nn.Linear(in_dim * 2 + (edge_dim if use_edge_features else 0), out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # Message transformation for bidirected edges
        self.bidirected_msg = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        W: torch.Tensor,
        M: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: (n_nodes, in_dim)
            W: Directed edge weights (n_nodes, n_nodes)
            M: Bidirected edge weights (n_nodes, n_nodes)
            edge_features: Optional (n_nodes, n_nodes, edge_dim)

        Returns:
            Updated node features (n_nodes, out_dim)
        """
        n_nodes = node_features.shape[0]

        # Self transformation
        h_self = self.node_mlp(node_features)

        # Aggregate messages from directed edges (parents)
        h_directed = torch.zeros_like(h_self)
        for i in range(n_nodes):
            # Find parents (nodes with edges pointing to i)
            parent_mask = W[:, i].abs() > EDGE_THRESHOLD
            if parent_mask.any():
                parent_features = node_features[parent_mask]
                target_features = node_features[i:i+1].expand(parent_features.shape[0], -1)

                if self.use_edge_features and edge_features is not None:
                    edge_feats = edge_features[parent_mask, i]
                    msg_input = torch.cat([parent_features, target_features, edge_feats], dim=-1)
                else:
                    msg_input = torch.cat([parent_features, target_features], dim=-1)

                messages = self.directed_msg(msg_input)
                weights = W[parent_mask, i].abs().unsqueeze(-1)
                h_directed[i] = (messages * weights).sum(dim=0)

        # Aggregate messages from bidirected edges (confounded)
        h_bidirected = torch.zeros_like(h_self)
        for i in range(n_nodes):
            neighbor_mask = M[i].abs() > EDGE_THRESHOLD
            neighbor_mask[i] = False  # Exclude self
            if neighbor_mask.any():
                neighbor_features = node_features[neighbor_mask]
                target_features = node_features[i:i+1].expand(neighbor_features.shape[0], -1)

                msg_input = torch.cat([neighbor_features, target_features], dim=-1)
                messages = self.bidirected_msg(msg_input)
                weights = M[i, neighbor_mask].abs().unsqueeze(-1)
                h_bidirected[i] = (messages * weights).sum(dim=0)

        # Combine all components
        h_out = h_self + h_directed + h_bidirected
        h_out = self.layer_norm(h_out)

        return h_out


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder for causal graphs.

    Produces embeddings at node, edge, and graph level.
    """

    def __init__(self, config: GraphEncoderConfig):
        super().__init__()
        self.config = config

        # Initial node embedding (for nodes without features)
        self.node_init = nn.Parameter(torch.randn(1, config.node_dim) * 0.1)

        # Edge feature encoder
        if config.use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(2, config.edge_dim),  # weight + sign
                nn.ReLU(),
                nn.Linear(config.edge_dim, config.edge_dim)
            )

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = config.node_dim if i == 0 else config.hidden_dim
            out_dim = config.hidden_dim
            self.layers.append(
                GraphConvLayer(
                    in_dim, out_dim, config.edge_dim, config.use_edge_features
                )
            )

        # Graph-level readout
        self.graph_readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        state: CausalState,
        node_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode causal graph.

        Args:
            state: CausalState with W and L matrices
            node_features: Optional initial node features (n_nodes, node_dim)

        Returns:
            Dict with 'node', 'edge', 'graph' embeddings
        """
        W = state.W
        M = state.M  # M = L @ L.T
        n_nodes = W.shape[0]

        # Initialize node features
        if node_features is None:
            h = self.node_init.expand(n_nodes, -1)
        else:
            h = node_features

        # Compute edge features
        edge_features = None
        if self.config.use_edge_features:
            edge_features = self._compute_edge_features(W, M)

        # Apply GNN layers
        for layer in self.layers:
            h = layer(h, W, M, edge_features)
            h = F.relu(h)
            h = self.dropout(h)

        # Node embeddings
        node_embeddings = h

        # Edge embeddings (for edges that exist)
        edge_embeddings = self._compute_edge_embeddings(node_embeddings, W, M)

        # Graph embedding via pooling
        if self.config.aggregation == "mean":
            graph_embedding = node_embeddings.mean(dim=0)
        elif self.config.aggregation == "sum":
            graph_embedding = node_embeddings.sum(dim=0)
        else:  # max
            graph_embedding = node_embeddings.max(dim=0)[0]

        graph_embedding = self.graph_readout(graph_embedding)

        return {
            "node": node_embeddings,
            "edge": edge_embeddings,
            "graph": graph_embedding
        }

    def _compute_edge_features(
        self,
        W: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge features from weight matrices."""
        n = W.shape[0]
        device = W.device

        # Stack weight and sign as edge features
        edge_feats = torch.zeros(n, n, 2, device=device)
        edge_feats[:, :, 0] = W.abs()
        edge_feats[:, :, 1] = torch.sign(W)

        # Add bidirected edge info
        edge_feats[:, :, 0] = edge_feats[:, :, 0] + M.abs() * 0.5

        # Encode through MLP
        edge_feats_flat = edge_feats.view(-1, 2)
        edge_encoded = self.edge_encoder(edge_feats_flat)
        return edge_encoded.view(n, n, -1)

    def _compute_edge_embeddings(
        self,
        node_embeddings: torch.Tensor,
        W: torch.Tensor,
        M: torch.Tensor
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Compute embeddings for each edge."""
        n = W.shape[0]
        edge_embeddings = {}

        # Directed edges
        for i in range(n):
            for j in range(n):
                if W[i, j].abs() > EDGE_THRESHOLD:
                    # Concatenate source and target node embeddings
                    edge_emb = torch.cat([node_embeddings[i], node_embeddings[j]])
                    edge_embeddings[(i, j)] = edge_emb

        # Bidirected edges (store once per pair)
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j].abs() > EDGE_THRESHOLD:
                    edge_emb = torch.cat([node_embeddings[i], node_embeddings[j]])
                    edge_embeddings[(i, j, "bidirected")] = edge_emb

        return edge_embeddings

    def encode_state(self, state: CausalState) -> torch.Tensor:
        """
        Get graph-level embedding for a causal state.

        Args:
            state: CausalState

        Returns:
            Graph embedding tensor
        """
        result = self.forward(state)
        return result["graph"]

    def encode_states(self, states: List[CausalState]) -> torch.Tensor:
        """
        Encode multiple causal states.

        Args:
            states: List of CausalState objects

        Returns:
            Tensor of shape (n_states, hidden_dim)
        """
        embeddings = []
        for state in states:
            emb = self.encode_state(state)
            embeddings.append(emb)
        return torch.stack(embeddings)


class StructuralFeatureExtractor:
    """
    Extract structural features from causal graphs.

    Computes hand-crafted features for baselines and interpretability.
    """

    def __init__(self):
        pass

    def extract_features(self, state: CausalState) -> Dict[str, float]:
        """Extract structural features from causal state."""
        W = state.W.detach().cpu().numpy()
        M = state.M.detach().cpu().numpy()
        n = W.shape[0]

        # Basic statistics (use FEATURE_EXTRACTION_THRESHOLD for robust counting)
        features = {
            "n_nodes": n,
            "n_directed_edges": (np.abs(W) > FEATURE_EXTRACTION_THRESHOLD).sum(),
            "n_bidirected_edges": ((np.abs(M) > FEATURE_EXTRACTION_THRESHOLD).sum() - n) / 2,  # Exclude diagonal
            "edge_density": (np.abs(W) > FEATURE_EXTRACTION_THRESHOLD).sum() / (n * (n - 1)),
            "avg_edge_weight": np.abs(W[np.abs(W) > FEATURE_EXTRACTION_THRESHOLD]).mean() if (np.abs(W) > FEATURE_EXTRACTION_THRESHOLD).any() else 0,
            "max_edge_weight": np.abs(W).max(),
        }

        # Degree statistics
        in_degrees = (np.abs(W) > FEATURE_EXTRACTION_THRESHOLD).sum(axis=0)
        out_degrees = (np.abs(W) > FEATURE_EXTRACTION_THRESHOLD).sum(axis=1)
        features["avg_in_degree"] = in_degrees.mean()
        features["avg_out_degree"] = out_degrees.mean()
        features["max_in_degree"] = in_degrees.max()
        features["max_out_degree"] = out_degrees.max()

        # Structural properties
        features["acyclicity"] = float(state.acyclicity_penalty().item())
        features["spectral_radius"] = float(state.spectral_radius().item())

        # Latent structure
        features["n_latent"] = state.n_latent
        features["latent_influence"] = np.abs(M).mean()

        return features

    def extract_batch_features(self, states: List[CausalState]) -> List[Dict[str, float]]:
        """Extract features from multiple states."""
        return [self.extract_features(s) for s in states]


class GraphSimilarity:
    """
    Compute similarity between causal graphs.

    Supports both embedding-based and structural similarity.
    """

    def __init__(self, encoder: Optional[GraphEncoder] = None):
        self.encoder = encoder

    def structural_similarity(
        self,
        state1: CausalState,
        state2: CausalState
    ) -> Dict[str, float]:
        """
        Compute structural similarity metrics.

        Returns:
            Dict with various similarity scores
        """
        W1 = state1.discretize()
        W2 = state2.discretize()

        # Ensure same size
        n1, n2 = W1.shape[0], W2.shape[0]
        if n1 != n2:
            raise ValueError(f"Graph sizes differ: {n1} vs {n2}")

        W1 = W1.detach().cpu().numpy()
        W2 = W2.detach().cpu().numpy()

        # Edge overlap
        edges1 = set(zip(*np.where(W1 != 0)))
        edges2 = set(zip(*np.where(W2 != 0)))

        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)

        return {
            "jaccard": intersection / union if union > 0 else 1.0,
            "precision": intersection / len(edges2) if edges2 else 1.0,
            "recall": intersection / len(edges1) if edges1 else 1.0,
            "shd": len(edges1 ^ edges2)  # Symmetric Hamming Distance
        }

    def embedding_similarity(
        self,
        state1: CausalState,
        state2: CausalState
    ) -> float:
        """
        Compute embedding-based similarity using GNN encoder.

        Returns:
            Cosine similarity score
        """
        if self.encoder is None:
            raise ValueError("GNN encoder required for embedding similarity")

        with torch.no_grad():
            emb1 = self.encoder.encode_state(state1)
            emb2 = self.encoder.encode_state(state2)

        # Cosine similarity
        similarity = F.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        )

        return float(similarity.item())


def create_graph_encoder(
    config: Optional[GraphEncoderConfig] = None
) -> GraphEncoder:
    """Factory function to create graph encoder."""
    config = config or GraphEncoderConfig()
    return GraphEncoder(config)
