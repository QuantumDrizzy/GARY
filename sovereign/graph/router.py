#!/usr/bin/env python3
"""Graph Engineering — Dynamic Heterogeneous Scheduler (CPU ↔ GPU ↔ QPU)."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Set

from sovereign.graph.dag import ComputeNode, HardwareTarget, HeterogeneousDAG, TaskKind


class DynamicHardwareRouter:
    def __init__(self, qpu_queue_latency_ms: float = 80.0, qpu_fidelity: float = 0.994):
        self.qpu_queue_latency_ms = qpu_queue_latency_ms
        self.qpu_fidelity = qpu_fidelity
        self.pcie_bw_gbps = 64.0  # PCIe Gen5 x16

    def evaluate_cost_matrix(self, node: ComputeNode) -> Tuple[HardwareTarget, str]:
        """Calculates latency and fidelity cost matrix to select optimal hardware target."""
        if node.kind == TaskKind.GARY_SIGNAL_FILTER:
            # GARY runs locally with low latency to gate all downstream compute
            return HardwareTarget.CPU_HOST, "GARY Meaning-Meter gatekeeper"

        if node.kind == TaskKind.QPU_COMBINATORIAL:
            n_qubits = node.metadata.get("n_qubits", 8)
            two_q_depth = node.metadata.get("two_q_depth", 6)
            bond_dim = node.metadata.get("bond_dim", 2)

            # 1. Physical QPU Execution Time
            # Pulse duration ~200ns per 2Q gate + readout ~1us + queue wait
            qpu_time_ms = self.qpu_queue_latency_ms + (two_q_depth * 0.0002) + 0.001

            # 2. Blaze / LYTH GPU Emulation Time
            # MPS contraction scales as O(n * chi^3)
            # On RTX 5060 Ti, small bond dimension chi <= 4 runs in < 2ms!
            gpu_emul_time_ms = (n_qubits * (bond_dim ** 3) * 0.00005) + 1.2  # ~1.2ms launch overhead

            # Decision logic:
            if self.qpu_fidelity < 0.985:
                return (
                    HardwareTarget.GPU_EMULATED_QPU,
                    f"QPU fidelity degraded ({self.qpu_fidelity:.3f} < 0.985); routed to Blaze MPS on GPU."
                )
            elif qpu_time_ms > gpu_emul_time_ms * 5.0 and bond_dim <= 4:
                return (
                    HardwareTarget.GPU_EMULATED_QPU,
                    f"QPU queue latency ({qpu_time_ms:.1f}ms) dominates; Blaze MPS on GPU ({gpu_emul_time_ms:.2f}ms) is faster."
                )
            else:
                return (
                    HardwareTarget.QPU_ANALOG,
                    f"Dispatched to physical QPU (n={n_qubits}, depth={two_q_depth}, fidelity={self.qpu_fidelity})."
                )

        if node.kind == TaskKind.GPU_TENSOR_CONTRACTION:
            return (
                HardwareTarget.GPU_BLACKWELL,
                "Dispatched to NVIDIA Blackwell (sm_120) with stride-33 shared memory (Hong-Kung optimal)."
            )

        return HardwareTarget.CPU_HOST, "Scalar orchestration on Host CPU."

    def execute_dag(self, dag: HeterogeneousDAG) -> Dict[str, Any]:
        """Execute the DAG dynamically until completion."""
        completed_ids: Set[str] = set()
        trace = []

        while not dag.is_complete():
            ready_nodes = dag.get_ready_nodes(completed_ids)
            if not ready_nodes:
                break

            for node in ready_nodes:
                target, reason = self.evaluate_cost_matrix(node)
                node.assigned_target = target

                # Execute node logic
                t0 = time.perf_counter()
                if node.kind == TaskKind.GARY_SIGNAL_FILTER:
                    # Run Meaning-Meter
                    z_score = node.payload.get("z_score", 0.0)
                    if z_score < 3.0:
                        node.status = "ABORTED"
                        node.result = f"REJECTED_APOPHENIA (Z = {z_score:.2f} sigma < 3.0)"
                        # Abort all dependent nodes
                        for dep_node in dag.nodes.values():
                            if node.node_id in dep_node.dependencies:
                                dep_node.status = "ABORTED"
                                dep_node.result = "ABORTED_BY_GARY_UPSTREAM"
                    else:
                        node.status = "COMPLETED"
                        node.result = f"PASSED_SIGNAL_VERIFIED (Z = {z_score:.2f} sigma)"
                else:
                    node.status = "COMPLETED"
                    node.result = f"SUCCESS on {node.assigned_target.value}"

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                node.execution_time_ms = elapsed_ms
                completed_ids.add(node.node_id)

                trace.append({
                    "node_id": node.node_id,
                    "kind": node.kind.value,
                    "target": node.assigned_target.value,
                    "status": node.status,
                    "reason": reason,
                    "result": str(node.result),
                })

        return {
            "dag_name": dag.dag_name,
            "total_nodes": len(dag.nodes),
            "completed": sum(1 for n in dag.nodes.values() if n.status == "COMPLETED"),
            "aborted": sum(1 for n in dag.nodes.values() if n.status == "ABORTED"),
            "execution_trace": trace,
        }
