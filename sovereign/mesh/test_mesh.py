#!/usr/bin/env python3
"""Tests for Mesh Engineering — Autonomous Hive Telemetry & Auto-Tuning."""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from sovereign.mesh.hive import AlertSeverity, SentinelMesh


def test_mesh_hive():
    mesh = SentinelMesh("Node-Blackwell-RTX5060Ti")

    # 1. Nominal check
    events = mesh.tick_hive()
    assert len(events) == 0, "Nominal state should have 0 alerts"
    assert mesh.telemetry.active_quantization_mode == "FP64_EXACT"

    # 2. Simulated thermal surge: GPU temp reaches 76 C
    mesh.update_telemetry(gpu_temp_celsius=76.0, gpu_vram_used_pct=78.0)
    events = mesh.tick_hive()
    assert len(events) == 1
    assert events[0].action_taken == "HOT_SWAP_TO_INT8_BLAZE_CORES"
    assert mesh.telemetry.active_quantization_mode == "INT8_CORE_QUANTIZED"

    # 3. Critical thermal surge: GPU temp reaches 82 C
    mesh.update_telemetry(gpu_temp_celsius=82.0)
    events = mesh.tick_hive()
    assert len(events) == 1
    assert events[0].action_taken == "HOT_SWAP_TO_INT4_BLAZE_CORES"
    assert mesh.telemetry.active_quantization_mode == "INT4_CORE_QUANTIZED"

    # 4. Quantum decoherence event: T2 drops to 18 us
    mesh.update_telemetry(qpu_t2_coherence_us=18.0)
    events = mesh.tick_hive()
    assert len(events) == 1
    assert events[0].initiator_sentinel == "QPUSentinel"
    assert "FALLBACK_MPS" in events[0].action_taken

    print("[MESH TEST] ALL TESTS PASSED. Autonomous sentinel hive auto-tuning verified.")


if __name__ == "__main__":
    test_mesh_hive()
