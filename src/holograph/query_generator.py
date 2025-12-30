"""
Query Generator for HOLOGRAPH.

Implements Expected Free Energy (EFE) based active query selection
for optimal information gathering from LLMs.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging

# Import centralized constants
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.config.constants import (
    QUERY_UNCERTAINTY_THRESHOLD,
    QUERY_MIN_EDGE_WEIGHT,
    DIRECTION_QUERY_WEIGHT,
    MECHANISM_QUERY_WEIGHT,
    CONFOUNDER_QUERY_WEIGHT,
    DEFAULT_LLM_CONFIDENCE,
)

from .causal_state import CausalState
from .llm_interface import BaseLLMInterface, LLMConfig, create_llm_interface

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Represents a causal query."""
    query_type: str  # 'edge', 'direction', 'mechanism', 'confounder'
    source: str  # Source variable name
    target: str  # Target variable name
    source_idx: int  # Source index
    target_idx: int  # Target index
    text: str  # Natural language query
    priority: float = 0.0  # EFE-based priority score
    answered: bool = False
    answer: Optional[Dict] = None


@dataclass
class QueryConfig:
    """Configuration for query generation."""
    max_queries_per_step: int = 5
    uncertainty_threshold: float = QUERY_UNCERTAINTY_THRESHOLD  # From constants
    min_edge_weight: float = QUERY_MIN_EDGE_WEIGHT  # From constants
    exploration_weight: float = 1.0  # Weight for epistemic value
    exploitation_weight: float = 1.0  # Weight for instrumental value
    use_semantic_similarity: bool = True


class QueryGenerator:
    """
    Generates and prioritizes queries using Expected Free Energy.

    EFE = Epistemic Value + Instrumental Value
    - Epistemic: Information gain about uncertain edges
    - Instrumental: Progress toward goal structure
    """

    def __init__(
        self,
        var_names: List[str],
        domain_context: str,
        config: Optional[QueryConfig] = None
    ):
        self.var_names = var_names
        self.n_vars = len(var_names)
        self.domain_context = domain_context
        self.config = config or QueryConfig()

        # Track query history
        self.query_history: List[Query] = []
        self.queried_edges: Set[Tuple[int, int]] = set()

        # Variable name to index mapping
        self.var_to_idx = {name: i for i, name in enumerate(var_names)}

    def generate_candidate_queries(
        self,
        state: CausalState
    ) -> List[Query]:
        """
        Generate all candidate queries based on current state.

        Args:
            state: Current causal state

        Returns:
            List of candidate Query objects
        """
        candidates = []
        W = state.W.detach().cpu().numpy()

        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i == j:
                    continue

                # Skip already queried edges
                if (i, j) in self.queried_edges:
                    continue

                source = self.var_names[i]
                target = self.var_names[j]

                # Generate different query types
                # 1. Edge existence query
                edge_query = Query(
                    query_type="edge",
                    source=source,
                    target=target,
                    source_idx=i,
                    target_idx=j,
                    text=f"Does {source} have a direct causal effect on {target}?"
                )
                candidates.append(edge_query)

                # 2. If edge might exist, ask about direction
                if abs(W[i, j]) > self.config.min_edge_weight:
                    direction_query = Query(
                        query_type="direction",
                        source=source,
                        target=target,
                        source_idx=i,
                        target_idx=j,
                        text=f"Is the causal effect from {source} to {target} positive (increasing) or negative (decreasing)?"
                    )
                    candidates.append(direction_query)

                # 3. Mechanism query for uncertain edges
                mechanism_query = Query(
                    query_type="mechanism",
                    source=source,
                    target=target,
                    source_idx=i,
                    target_idx=j,
                    text=f"What is the mechanism by which {source} affects {target}?"
                )
                candidates.append(mechanism_query)

        # 4. Confounder queries for pairs with potential bidirected edges
        M = state.M.detach().cpu().numpy()
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if abs(M[i, j]) > self.config.min_edge_weight:
                    source = self.var_names[i]
                    target = self.var_names[j]
                    confounder_query = Query(
                        query_type="confounder",
                        source=source,
                        target=target,
                        source_idx=i,
                        target_idx=j,
                        text=f"Is there a common cause (confounder) affecting both {source} and {target}?"
                    )
                    candidates.append(confounder_query)

        return candidates

    def compute_efe(
        self,
        query: Query,
        state: CausalState,
        beliefs: Optional[Dict] = None
    ) -> float:
        """
        Compute Expected Free Energy for a query.

        EFE = E[KL(Q(s'|o,a)||P(s'|a))] + E[log P(o|s',a)]
            ≈ Epistemic Value + Instrumental Value

        Args:
            query: Query to evaluate
            state: Current causal state
            beliefs: Current edge beliefs (optional)

        Returns:
            EFE score (lower is better, we negate for priority)
        """
        i, j = query.source_idx, query.target_idx
        W = state.W.detach().cpu().numpy()
        M = state.M.detach().cpu().numpy()

        # Epistemic value: Information gain about uncertain edges
        # Use entropy as proxy for uncertainty
        edge_weight = abs(W[i, j])
        # Map weight to uncertainty: high uncertainty when weight near 0.5
        uncertainty = 1.0 - abs(edge_weight - 0.5) * 2
        uncertainty = max(0, min(1, uncertainty))

        # For bidirected edges
        if query.query_type == "confounder":
            bidirected_weight = abs(M[i, j])
            uncertainty = 1.0 - abs(bidirected_weight - 0.5) * 2
            uncertainty = max(0, min(1, uncertainty))

        epistemic_value = uncertainty

        # Instrumental value: How much this query reduces expected loss
        # Edges with higher current weights have more impact on loss
        if query.query_type == "edge":
            instrumental_value = 1.0 - edge_weight  # Prioritize uncertain edges
        elif query.query_type == "direction":
            instrumental_value = edge_weight * DIRECTION_QUERY_WEIGHT  # Only valuable if edge exists
        elif query.query_type == "mechanism":
            instrumental_value = edge_weight * MECHANISM_QUERY_WEIGHT  # Context is less critical
        else:  # confounder
            instrumental_value = abs(M[i, j]) * CONFOUNDER_QUERY_WEIGHT

        # Combine with configurable weights
        efe = (
            self.config.exploration_weight * epistemic_value +
            self.config.exploitation_weight * instrumental_value
        )

        return efe

    def prioritize_queries(
        self,
        candidates: List[Query],
        state: CausalState,
        beliefs: Optional[Dict] = None
    ) -> List[Query]:
        """
        Prioritize queries by EFE score.

        Args:
            candidates: List of candidate queries
            state: Current causal state
            beliefs: Current edge beliefs

        Returns:
            Sorted list of queries (highest priority first)
        """
        for query in candidates:
            query.priority = self.compute_efe(query, state, beliefs)

        # Sort by priority (descending)
        sorted_queries = sorted(candidates, key=lambda q: q.priority, reverse=True)

        return sorted_queries

    def select_queries(
        self,
        state: CausalState,
        max_queries: Optional[int] = None
    ) -> List[Query]:
        """
        Select top queries for current step.

        Args:
            state: Current causal state
            max_queries: Maximum number of queries to select

        Returns:
            List of selected queries
        """
        max_queries = max_queries or self.config.max_queries_per_step

        # Generate candidates
        candidates = self.generate_candidate_queries(state)

        # Prioritize by EFE
        prioritized = self.prioritize_queries(candidates, state)

        # P0: Filter by uncertainty threshold - only select queries with priority above threshold
        # This ensures we don't waste queries on low-information edges
        filtered = [
            q for q in prioritized
            if q.priority >= self.config.uncertainty_threshold
        ]

        # Log filtering effect
        if len(prioritized) > 0 and len(filtered) < len(prioritized):
            logger.debug(
                f"Filtered {len(prioritized) - len(filtered)} queries below "
                f"uncertainty threshold {self.config.uncertainty_threshold}"
            )

        # Select top queries from filtered list
        selected = filtered[:max_queries]

        # Mark as queried
        for query in selected:
            self.queried_edges.add((query.source_idx, query.target_idx))

        return selected

    def format_query_for_llm(
        self,
        query: Query,
        include_context: bool = True
    ) -> str:
        """
        Format query for LLM submission.

        Args:
            query: Query object
            include_context: Whether to include domain context

        Returns:
            Formatted query string
        """
        prompt_parts = []

        if include_context:
            prompt_parts.append(f"Domain: {self.domain_context}")
            prompt_parts.append("")

        prompt_parts.append(f"Question: {query.text}")
        prompt_parts.append("")
        prompt_parts.append("Please provide:")
        prompt_parts.append("1. Your answer (yes/no/uncertain)")
        prompt_parts.append("2. Confidence level (0.0-1.0)")
        prompt_parts.append("3. Brief reasoning")

        if query.query_type == "edge":
            prompt_parts.append("4. Estimated effect strength (-1.0 to 1.0, 0 if no effect)")

        return "\n".join(prompt_parts)

    def process_llm_response(
        self,
        query: Query,
        response: Dict
    ) -> Query:
        """
        Process LLM response and update query.

        Args:
            query: Original query
            response: LLM response dict

        Returns:
            Updated query with answer
        """
        query.answered = True
        query.answer = response
        self.query_history.append(query)
        return query

    def get_edge_update(
        self,
        query: Query
    ) -> Optional[Tuple[int, int, float]]:
        """
        Convert query answer to edge weight update.

        Args:
            query: Answered query

        Returns:
            Tuple of (source_idx, target_idx, weight_update) or None
        """
        if not query.answered or query.answer is None:
            return None

        answer = query.answer
        i, j = query.source_idx, query.target_idx

        if query.query_type == "edge":
            # Direct edge weight from answer
            if answer.get("answer") == "no":
                return (i, j, 0.0)
            else:
                weight = answer.get("edge_weight", 0.5)
                confidence = answer.get("confidence", 0.5)
                return (i, j, weight * confidence)

        elif query.query_type == "direction":
            # Direction affects sign
            direction = answer.get("answer", "positive")
            sign = 1.0 if direction == "positive" else -1.0
            return (i, j, sign)

        return None


class ActiveQueryAgent:
    """
    Active learning agent that queries LLM for causal information.

    Combines QueryGenerator with LLM interface for end-to-end querying.
    """

    def __init__(
        self,
        var_names: List[str],
        domain_context: str,
        llm_interface: BaseLLMInterface,
        config: Optional[QueryConfig] = None
    ):
        self.generator = QueryGenerator(var_names, domain_context, config)
        self.llm = llm_interface
        self.total_queries = 0

    def query_step(
        self,
        state: CausalState,
        max_queries: int = 5
    ) -> Tuple[List[Query], List[Tuple[int, int, float]]]:
        """
        Perform one step of active querying.

        Args:
            state: Current causal state
            max_queries: Number of queries to ask

        Returns:
            Tuple of (answered queries, edge updates)
        """
        # Select queries
        queries = self.generator.select_queries(state, max_queries)

        # Ask LLM
        answered_queries = []
        edge_updates = []

        for query in queries:
            prompt = self.generator.format_query_for_llm(query)

            try:
                response = self.llm.answer_causal_query(
                    query.text,
                    self.generator.domain_context
                )

                query = self.generator.process_llm_response(query, response)
                answered_queries.append(query)

                update = self.generator.get_edge_update(query)
                if update:
                    edge_updates.append(update)

                self.total_queries += 1

            except Exception as e:
                logger.warning(f"Query failed: {e}")
                continue

        return answered_queries, edge_updates

    def batch_query(
        self,
        state: CausalState,
        queries: List[Query]
    ) -> List[Query]:
        """
        Submit batch of queries to LLM.

        Args:
            state: Current causal state
            queries: List of queries to ask

        Returns:
            List of answered queries
        """
        answered = []

        for query in queries:
            prompt = self.generator.format_query_for_llm(query)

            try:
                response = self.llm.answer_causal_query(
                    query.text,
                    self.generator.domain_context
                )

                query = self.generator.process_llm_response(query, response)
                answered.append(query)
                self.total_queries += 1

            except Exception as e:
                logger.warning(f"Query failed: {e}")

        return answered

    def get_statistics(self) -> Dict:
        """Get query statistics."""
        type_counts = {}
        for query in self.generator.query_history:
            qtype = query.query_type
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        return {
            "total_queries": self.total_queries,
            "unique_edges_queried": len(self.generator.queried_edges),
            "queries_by_type": type_counts
        }


def create_query_generator(
    var_names: List[str],
    domain_context: str,
    config: Optional[QueryConfig] = None
) -> QueryGenerator:
    """Factory function to create query generator."""
    return QueryGenerator(var_names, domain_context, config)


def create_active_query_agent(
    var_names: List[str],
    domain_context: str,
    llm_config: Optional[LLMConfig] = None,
    query_config: Optional[QueryConfig] = None
) -> ActiveQueryAgent:
    """Factory function to create active query agent."""
    llm = create_llm_interface(llm_config or LLMConfig())
    return ActiveQueryAgent(var_names, domain_context, llm, query_config)
