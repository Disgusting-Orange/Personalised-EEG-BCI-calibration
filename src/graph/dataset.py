"""PyTorch Geometric InMemoryDataset wrapper for Subject-Level EEG Motor Imagery Regression.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import torch
from torch_geometric.data import Data, InMemoryDataset

from src.graph.builder import build_subject_graph

logger = logging.getLogger("graph.dataset")


class EEGGraphDataset(InMemoryDataset):
    """PyTorch Geometric Dataset for EEG Motor Imagery Continuous Regression."""

    def __init__(
        self,
        root: Union[str, Path],
        subjects: Sequence[str] | None = None,
        config: dict[str, Any] | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self.root_dir = Path(root)
        self.subjects = list(subjects) if subjects else []
        self.config = config or {}

        super().__init__(str(self.root_dir), transform, pre_transform, pre_filter)

        # Load processed collated dataset if exists
        processed_path = self.root_dir / "pyg_dataset.pt"
        rebuild = bool(config.get("force_rebuild", False)) if config else False
        if processed_path.exists() and not rebuild:
            self.data, self.slices = torch.load(processed_path, weights_only=False)
        else:
            self._process_and_save()

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["pyg_dataset.pt"]

    def download(self) -> None:
        pass

    def _process_and_save(self) -> None:
        """Build graph objects for all subjects, collate, and save to root directory."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        data_list: list[Data] = []
        subject_manifest: list[dict[str, Any]] = []

        for sub_id in self.subjects:
            try:
                g_data = build_subject_graph(sub_id, self.config)
                data_list.append(g_data)
                # Save individual PyG Data object
                torch.save(g_data, self.root_dir / f"{sub_id}_graph.pt")
                subject_manifest.append(g_data.metadata)
            except Exception as exc:
                logger.error("Failed to build graph for subject %s: %s", sub_id, exc)
                raise exc

        if not data_list:
            raise ValueError("No graph objects were successfully built for dataset.")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.root_dir / "pyg_dataset.pt")

        # Save JSON manifest
        manifest_path = self.root_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(subject_manifest, indent=2), encoding="utf-8")

        self.data, self.slices = data, slices
        logger.info("Successfully collated and saved EEGGraphDataset (%d graphs) at %s", len(data_list), self.root_dir)
