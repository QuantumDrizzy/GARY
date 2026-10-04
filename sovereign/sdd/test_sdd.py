#!/usr/bin/env python3
"""Tests for the SDD Hardware & Quantum Contract Validator."""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from sovereign.sdd.validator import SDDValidator


def test_sdd_pipeline():
    validator = SDDValidator()

    # 1. Compliant CUDA Kernel (Stride 33, 48 registers, high FMA intensity)
    valid_kernel = """
    .version 8.0
    .target sm_120
    .entry matmul_kernel() {
        .reg .b32 %r<48>;
        .shared .f32 tile[32][33];  // Stride 33: Bank conflict free
        ld.shared.f32 %r1, [tile + 0];
        fma.rn.f32 %r2, %r1, %r1, %r3;
        fma.rn.f32 %r4, %r2, %r1, %r3;
        st.global.f32 [global_out], %r4;
    }
    """
    res_cuda = validator.validate_cuda_kernel(valid_kernel)
    assert res_cuda["passed"], f"Expected pass, got violations: {res_cuda['violations']}"

    # 2. Non-Compliant CUDA Kernel (Stride 32 bank conflict, 80 registers)
    bad_kernel = """
    .version 8.0
    .target sm_120
    .entry bad_kernel() {
        .reg .b32 %r<80>;          // Exceeds 64 limit
        .shared .f32 tile[32][32]; // Stride 32: 32-way bank conflict!
        if (threadIdx.x % 32 == 0) { // Warp divergence!
            ld.shared.f32 %r1, [tile];
        }
    }
    """
    res_bad = validator.validate_cuda_kernel(bad_kernel)
    assert not res_bad["passed"], "Expected failure for bad kernel"
    assert len(res_bad["violations"]) >= 2

    # 3. Compliant Quantum Circuit (Within T2 depth, native gates, coupled edges)
    valid_circuit = [
        {"gate": "h", "qubits": [0]},
        {"gate": "cz", "qubits": [0, 1]},  # Edge [0, 1] exists
        {"gate": "rx", "qubits": [1]},
        {"gate": "cz", "qubits": [1, 2]},  # Edge [1, 2] exists
        {"gate": "rz", "qubits": [2]},
    ]
    res_q = validator.validate_quantum_circuit(valid_circuit)
    assert res_q["passed"], f"Expected pass, got: {res_q['violations']}"

    # 4. Generate Immutable Certificate
    cert = validator.generate_certificate(res_cuda, res_q)
    assert cert["status"] == "CERTIFIED_FOR_METAL"
    assert len(cert["certificate_sha256"]) == 64

    print("[SDD TEST] ALL TESTS PASSED. Hardware contract enforcement verified.")


if __name__ == "__main__":
    test_sdd_pipeline()
