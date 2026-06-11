"""
DeCal: Decoupled Calibration for Drift-Aware Budget Control in LLM Serving.

A framework that monitors workload drift in LLM serving and adaptively adjusts
per-window budget targets using decoupled empirical calibration. The current
implementation instantiates the scalar budget as KV-cache capacity.

Three core pillars:
    1. Online drift monitoring over scalar workload summaries
    2. Drift-adaptive per-window budget control
    3. Cross-client distribution transfer for cold-start and recovery
"""

__version__ = "0.1.0"

from decal.detectors import (
    PageHinkleyDetector,
    CUSUMDetector,
    EWMADetector,
    ADWINDetector,
    KSWINDetector,
)
from decal.allocators import (
    BaseAllocator,
    AllocationResult,
    DriftAdaptiveAllocator,
)
from decal.transfer import (
    WassersteinTransfer,
    ClientSimilarityGraph,
)
from decal.orchestrator import DeCalOrchestrator, OrchestratorConfig
from decal.evaluation import Evaluator, Metrics, WindowResult
from decal.data_loader import (
    ServeGenDataLoader,
    BurstGPTDataLoader,
    preprocess_for_decal,
    preprocess_burstgpt_for_decal,
)

__all__ = [
    # Detectors
    "PageHinkleyDetector",
    "CUSUMDetector",
    "EWMADetector",
    "ADWINDetector",
    "KSWINDetector",
    # Allocator base
    "BaseAllocator",
    "AllocationResult",
    # DeCal core
    "DriftAdaptiveAllocator",
    # Transfer
    "WassersteinTransfer",
    "ClientSimilarityGraph",
    # Orchestrator
    "DeCalOrchestrator",
    "OrchestratorConfig",
    # Evaluation
    "Evaluator",
    "Metrics",
    "WindowResult",
    # Data Loading — ServeGen
    "ServeGenDataLoader",
    "preprocess_for_decal",
    # Data Loading — BurstGPT
    "BurstGPTDataLoader",
    "preprocess_burstgpt_for_decal",
]
