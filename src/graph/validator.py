"""Validation module for checking graph integrity, dimensions, degree distribution, and numeric stability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import torch
from torch_geometric.data import Data

logger = logging.getLogger("graph.validator")


def validate_graph(data: Data, expected_nodes: int = 64) -> dict[str, Any]:
    """Validate a single PyTorch Geometric Data graph object for integrity and numeric stability.

    Parameters
    ----------
    data:
        PyG Data graph object.
    expected_nodes:
        Expected number of nodes (default 64).

    Returns
    -------
    dict[str, Any]:
        Validation result details and boolean status.
    """
    errors: list[str] = []

    # 1. Node count check
    if data.num_nodes != expected_nodes:
        errors.append(f"Invalid num_nodes {data.num_nodes}, expected {expected_nodes}")

    if data.x.shape[0] != expected_nodes:
        errors.append(f"Node feature rows {data.x.shape[0]} != {expected_nodes}")

    # 2. Edge tensor shape check
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        errors.append(f"Invalid edge_index shape {data.edge_index.shape}, expected (2, E)")

    if data.edge_weight.ndim != 1 or data.edge_weight.shape[0] != data.edge_index.shape[1]:
        errors.append(f"Edge weight length {data.edge_weight.shape} != edge count {data.edge_index.shape[1]}")

    # 3. Numeric stability check (No NaN / Inf)
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        errors.append("NaN or Inf detected in node features x")

    if torch.isnan(data.edge_weight).any() or torch.isinf(data.edge_weight).any():
        errors.append("NaN or Inf detected in edge weights")

    if torch.isnan(data.y).any() or torch.isinf(data.y).any():
        errors.append("NaN or Inf detected in target y")

    # 4. Target bounds check
    y_val = float(data.y.item())
    if y_val < 0.0 or y_val > 1.0:
        errors.append(f"Target value {y_val} outside [0.0, 1.0] range")

    # 5. Non-isolated nodes check
    src_nodes = data.edge_index[0]
    degrees = torch.bincount(src_nodes, minlength=expected_nodes)
    isolated_nodes = (degrees == 0).nonzero(as_tuple=True)[0].tolist()
    if isolated_nodes:
        errors.append(f"Isolated nodes detected with degree 0: {isolated_nodes}")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "subject_id": getattr(data, "subject_id", "unknown"),
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "feature_dim": data.x.shape[1],
        "target_y": y_val,
        "isolated_node_count": len(isolated_nodes),
        "errors": errors,
    }


def validate_graph_dataset_directory(
    dataset_dir: Union[str, Path],
    expected_subjects: int | None = None,
) -> dict[str, Any]:
    """Validate all graph artifacts inside a graph dataset directory.

    Parameters
    ----------
    dataset_dir:
        Directory path containing PyG graphs.
    expected_subjects:
        Optional expected subject count.

    Returns
    -------
    dict[str, Any]:
        Cohort validation summary dictionary.
    """
    dataset_path = Path(dataset_dir)
    graph_files = sorted(list(dataset_path.glob("*_graph.pt")))

    if expected_subjects and len(graph_files) != expected_subjects:
        logger.warning("Found %d graph files, expected %d", len(graph_files), expected_subjects)

    passed_count = 0
    failed_count = 0
    sub_results: list[dict[str, Any]] = []

    for gf in graph_files:
        g_data = torch.load(gf, weights_only=False)
        v_res = validate_graph(g_data)
        sub_results.append(v_res)
        if v_res["status"] == "PASS":
            passed_count += 1
        else:
            failed_count += 1

    manifest_file = dataset_path / "dataset_manifest.json"
    manifest_exists = manifest_file.exists()
    pyg_dataset_exists = (dataset_path / "pyg_dataset.pt").exists()

    overall_status = "PASS" if (failed_count == 0 and pyg_dataset_exists and manifest_exists) else "FAIL"

    summary = {
        "overall_status": overall_status,
        "dataset_dir": str(dataset_path),
        "total_graphs_evaluated": len(graph_files),
        "passed_graphs": passed_count,
        "failed_graphs": failed_count,
        "manifest_exists": manifest_exists,
        "pyg_collated_exists": pyg_dataset_exists,
        "graph_details": sub_results,
    }

    report_path = dataset_path / "validation_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Validated graph dataset at %s: status=%s (%d/%d passed)", dataset_path, overall_status, passed_count, len(graph_files))

    return summary
