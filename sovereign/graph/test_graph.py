#!/usr/bin/env python3
"""Tests for Graph Engineering — Dynamic Heterogeneous Scheduling."""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from sovereign.graph.dag import ComputeNode, HardwareTarget, HeterogeneousDAG, TaskKind
from sovereign.graph.router import DynamicHardwareRouter


def test_graph_routing():
    # 1. Pipeline with Real Signal (Z = 757.7 sigma)
    dag_real = HeterogeneousDAG("QuBLAR-Blaze-Quantum-Pipeline")

    n1 = dag_real.add_node(ComputeNode(
        node_id="gatekeeper",
        kind=TaskKind.GARY_SIGNAL_FILTER,
        payload={"z_score": 757.73}
    ))
    n2 = dag_real.add_node(ComputeNode(
        node_id="combinatorial_ising",
        kind=TaskKind.QPU_COMBINATORIAL,
        payload={"branches": 2**20},
        dependencies=["gatekeeper"],
        metadata={"n_qubits": 12, "two_q_depth": 8, "bond_dim": 2}
    ))
    n3 = dag_real.add_node(ComputeNode(
        node_id="tt_mps_compression",
        kind=TaskKind.GPU_TENSOR_CONTRACTION,
        payload={"shape": [2]*12},
        dependencies=["combinatorial_ising"],
        metadata={"precision": "int8"}
    ))

    router = DynamicHardwareRouter(qpu_queue_latency_ms=120.0, qpu_fidelity=0.995)
    res_real = router.execute_dag(dag_real)

    assert res_real["completed"] == 3
    assert res_real["aborted"] == 0
    # Because QPU queue is 120ms and bond_dim=2, it auto-routed to GPU_EMULATED_QPU!
    assert dag_real.nodes["combinatorial_ising"].assigned_target == HardwareTarget.GPU_EMULATED_QPU
    assert dag_real.nodes["tt_mps_compression"].assigned_target == HardwareTarget.GPU_BLACKWELL

    # 2. Pipeline with Apophenia Noise (Z = 1.1 sigma)
    dag_noise = HeterogeneousDAG("Spurious-Noise-Pipeline")
    dag_noise.add_node(ComputeNode(
        node_id="gatekeeper",
        kind=TaskKind.GARY_SIGNAL_FILTER,
        payload={"z_score": 1.10}  # Below 3.0 sigma!
    ))
    dag_noise.add_node(ComputeNode(
        node_id="expensive_gpu_job",
        kind=TaskKind.GPU_TENSOR_CONTRACTION,
        payload={},
        dependencies=["gatekeeper"]
    ))

    res_noise = router.execute_dag(dag_noise)
    assert res_noise["aborted"] == 2  # GARY rejected + downstream dropped!
    assert dag_noise.nodes["expensive_gpu_job"].status == "ABORTED"

    print("[GRAPH TEST] ALL TESTS PASSED. Dynamic heterogeneous routing & fallbacks verified.")


if __name__ == "__main__":
    test_graph_routing()
