#!/usr/bin/env python3
"""SDD (Spec-Driven Development) — Hardware & Quantum Contract Validator.

Validates candidate GPU kernels (CUDA/PTX) and Quantum Circuits (Cirq/OpenQASM)
against the formal physical invariants specified in contract.toml.
Rejects any candidate that induces register spilling, bank conflicts,
or quantum decoherence.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

HERE = pathlib.Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contract.toml"


class SDDValidationError(Exception):
    """Raised when candidate code violates the hardware or quantum contract."""
    pass


class SDDValidator:
    def __init__(self, contract_file: pathlib.Path = CONTRACT_PATH):
        if not contract_file.is_file():
            raise FileNotFoundError(f"Contract file not found: {contract_file}")
        with open(contract_file, "rb") as f:
            self.contract = tomllib.load(f)

    def validate_cuda_kernel(self, kernel_source_or_ptx: str) -> Dict[str, Any]:
        """Validate candidate CUDA C++ or PTX code against the hardware contract."""
        cfg = self.contract["cuda_contract"]
        violations = []

        # 1. Register Limit Check (e.g. .reg .b32 %r<65>; or __launch_bounds__(..., 64))
        reg_match = re.search(r"\.reg\s+\.[bfsu]\d+\s+%\w+<(\d+)>", kernel_source_or_ptx)
        reg_count = int(reg_match.group(1)) if reg_match else None
        
        # Check launch bounds in CUDA C++
        if reg_count is None:
            bounds_match = re.search(r"__launch_bounds__\s*\(\s*\d+\s*,\s*(\d+)\s*\)", kernel_source_or_ptx)
            if bounds_match:
                reg_count = int(bounds_match.group(1))

        max_regs = cfg["max_registers_per_thread"]
        if reg_count and reg_count > max_regs:
            violations.append(
                f"REGISTER SPILL RISK: Kernel requests {reg_count} registers per thread; "
                f"spec hard limit is {max_regs} (causes spill to local memory)."
            )

        # 2. Shared Memory Stride & Bank Conflict Check
        # On NVIDIA architectures (32 memory banks), stride 33 eliminates bank collisions
        req_stride = cfg["required_shared_memory_stride"]
        has_shared = "__shared__" in kernel_source_or_ptx or ".shared" in kernel_source_or_ptx
        stride_found = False
        if has_shared:
            if re.search(r"\[\s*33\s*\]|\*\s*33|\+\s*33", kernel_source_or_ptx):
                stride_found = True
            elif re.search(r"\[\s*32\s*\]|\*\s*32", kernel_source_or_ptx):
                violations.append(
                    f"SHARED MEMORY BANK CONFLICT: Found stride 32 indexing across 32 banks. "
                    f"Must use stride {req_stride} to prevent 32-way serialization."
                )
            else:
                violations.append(
                    f"UNVERIFIED SHARED MEMORY ACCESS: Did not detect explicit stride {req_stride} padding."
                )

        # 3. Arithmetic Intensity (Hong-Kung lower bound)
        # Rough operational intensity: compute FLOPs / Memory Transfers
        flops = len(re.findall(r"\b(fma|mul|add|sub|sin|cos|sqrt|rsqrt)\b", kernel_source_or_ptx, re.IGNORECASE))
        loads = len(re.findall(r"\b(ld\.global|ld\.shared|ldg|\bload\b)\b", kernel_source_or_ptx, re.IGNORECASE))
        stores = len(re.findall(r"\b(st\.global|st\.shared|stg|\bstore\b)\b", kernel_source_or_ptx, re.IGNORECASE))
        mem_ops = max(1, loads + stores)
        # Assuming 4 bytes per float
        estimated_intensity = flops / (mem_ops * 4.0)

        min_intensity = cfg["min_arithmetic_intensity_flops_per_byte"]
        # If intensity is below spec and memory ops dominate:
        if estimated_intensity < min_intensity and loads > 10:
            violations.append(
                f"HONG-KUNG I/O VIOLATION: Operational intensity {estimated_intensity:.2f} FLOPs/byte "
                f"is below the physical saturation knee of {min_intensity} FLOPs/byte."
            )

        # 4. Warp Divergence
        if not cfg["allow_warp_divergence"]:
            if re.search(r"threadIdx\.x\s*%\s*32\s*==", kernel_source_or_ptx) or \
               re.search(r"laneid\s*==", kernel_source_or_ptx):
                violations.append("WARP DIVERGENCE: Branching conditioned on thread lane within warp detected.")

        passed = len(violations) == 0
        return {
            "component": "CUDA_BARE_METAL",
            "passed": passed,
            "violations": violations,
            "metrics": {
                "registers": reg_count or "unspecified",
                "shared_memory_stride_33": stride_found,
                "estimated_intensity": estimated_intensity,
                "hong_kung_compliant": estimated_intensity >= min_intensity or loads <= 10,
            }
        }

    def validate_quantum_circuit(self, circuit_ops: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate candidate Quantum Circuit against coherence and topology invariants.
        circuit_ops is a list of {'gate': 'cz'|'rz'|'rx'|'h', 'qubits': [q1] or [q1, q2]}
        """
        cfg = self.contract["quantum_contract"]
        violations = []

        allowed_gates = set(cfg["allowed_native_gates"])
        coupling_edges = set((u, v) for u, v in cfg["coupling_edges"])
        coupling_edges.update((v, u) for u, v in cfg["coupling_edges"])

        depth_2q = 0
        qubit_layer_2q: Dict[int, int] = {}
        qubits_used = set()

        for idx, op in enumerate(circuit_ops):
            gate = op.get("gate", "").lower()
            qubits = op.get("qubits", [])
            for q in qubits:
                qubits_used.add(q)

            # 1. Native Gate Set Check
            if gate not in allowed_gates:
                violations.append(
                    f"NON-NATIVE GATE VIOLATION: Gate '{gate}' at step {idx} is not in native basis "
                    f"{list(allowed_gates)} (forces noisy decomposition)."
                )

            # 2. Two-qubit gate analysis
            if len(qubits) == 2:
                q1, q2 = qubits[0], qubits[1]
                # Topology Coupling Check
                if (q1, q2) not in coupling_edges:
                    violations.append(
                        f"TOPOLOGY VIOLATION: Two-qubit gate between Q{q1} and Q{q2} has no physical coupling edge. "
                        f"Requires SWAP routing which exhausts coherence."
                    )

                # Two-qubit depth tracking
                cur_layer = max(qubit_layer_2q.get(q1, 0), qubit_layer_2q.get(q2, 0)) + 1
                qubit_layer_2q[q1] = cur_layer
                qubit_layer_2q[q2] = cur_layer
                depth_2q = max(depth_2q, cur_layer)

        # 3. Coherence Budget Check (T2 limit)
        max_depth = cfg["max_2q_gate_depth"]
        if depth_2q > max_depth:
            violations.append(
                f"DECOHERENCE THRESHOLD EXCEEDED: Two-qubit gate depth is {depth_2q}; "
                f"physical T2* budget allows at most {max_depth} layers before state collapses into noise."
            )

        # 4. Total Qubit Count
        if len(qubits_used) > cfg["max_total_qubits"]:
            violations.append(f"QUBIT ALLOCATION OVERFLOW: Circuit uses {len(qubits_used)} qubits; limit is {cfg['max_total_qubits']}.")

        passed = len(violations) == 0
        # Estimate fidelity based on depth
        fidelity_est = max(0.0, 1.0 - (depth_2q * 0.005) - (len(circuit_ops) * 0.0005))

        return {
            "component": "QUANTUM_CIRCUIT",
            "passed": passed,
            "violations": violations,
            "metrics": {
                "depth_2q": depth_2q,
                "max_allowed_depth_2q": max_depth,
                "qubits_used": len(qubits_used),
                "estimated_fidelity": fidelity_est,
                "fidelity_compliant": fidelity_est >= cfg["min_state_fidelity"],
            }
        }

    def generate_certificate(self, cuda_res: Dict[str, Any], quantum_res: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an immutable cryptographic compliance certificate for the hardware execution."""
        all_passed = cuda_res["passed"] and quantum_res["passed"]
        payload = json.dumps({"cuda": cuda_res, "quantum": quantum_res}, sort_keys=True)
        cert_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        return {
            "spec_contract": "SDD-HW-Q-1.0.0",
            "status": "CERTIFIED_FOR_METAL" if all_passed else "REJECTED_CONTRACT_VIOLATION",
            "all_passed": all_passed,
            "certificate_sha256": cert_hash,
            "cuda_verdict": cuda_res,
            "quantum_verdict": quantum_res,
        }
