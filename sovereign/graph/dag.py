#!/usr/bin/env python3
"""Graph Engineering — Heterogeneous Computation DAG Definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TaskKind(str, Enum):
    GARY_SIGNAL_FILTER = "GARY_SIGNAL_FILTER"          # Epistemic meaning-meter & apophenia firewall
    QPU_COMBINATORIAL = "QPU_COMBINATORIAL"            # Rydberg / Ising / MIS combinatorial problem
    GPU_TENSOR_CONTRACTION = "GPU_TENSOR_CONTRACTION"  # High-order tensor-train / TT-SVD (Blaze / LYTH)
    CPU_SCALAR_CONTROL = "CPU_SCALAR_CONTROL"          # State synchronization & classical reductions


class HardwareTarget(str, Enum):
    UNASSIGNED = "UNASSIGNED"
    CPU_HOST = "CPU_HOST"
    GPU_BLACKWELL = "GPU_BLACKWELL"
    QPU_ANALOG = "QPU_ANALOG"
    GPU_EMULATED_QPU = "GPU_EMULATED_QPU"  # Fallback to Blaze/LYTH MPS simulation
    ABORTED = "ABORTED"                    # Dropped by GARY firewall


@dataclass
class ComputeNode:
    node_id: str
    kind: TaskKind
    payload: Any
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    assigned_target: HardwareTarget = HardwareTarget.UNASSIGNED
    execution_time_ms: float = 0.0
    status: str = "PENDING"
    result: Any = None


class HeterogeneousDAG:
    def __init__(self, dag_name: str = "Hybrid-HPC-DAG"):
        self.dag_name = dag_name
        self.nodes: Dict[str, ComputeNode] = {}

    def add_node(self, node: ComputeNode) -> ComputeNode:
        if node.node_id in self.nodes:
            raise ValueError(f"Node already exists: {node.node_id}")
        self.nodes[node.node_id] = node
        return node

    def get_ready_nodes(self, completed_ids: Set[str]) -> List[ComputeNode]:
        ready = []
        for n in self.nodes.values():
            if n.node_id not in completed_ids and n.status == "PENDING":
                if all(dep in completed_ids for dep in n.dependencies):
                    ready.append(n)
        return ready

    def is_complete(self) -> bool:
        return all(n.status in ("COMPLETED", "ABORTED", "FAILED") for n in self.nodes.values())
