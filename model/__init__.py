"""
SignalScope Forensics Model Package
Core detection, calibration, explainability, and metadata inspection modules.
"""

from .backbone import SignalScopeDetector
from .calibration import TemperatureCalibrator
from .explainer import LayerCAMExplainer
from .metadata_inspector import MetadataInspector

__all__ = [
    "SignalScopeDetector",
    "TemperatureCalibrator",
    "LayerCAMExplainer",
    "MetadataInspector",
]
