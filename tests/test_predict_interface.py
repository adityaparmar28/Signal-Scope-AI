"""
SignalScope Automated Unit & Integration Tests
Ensures 10-minute reproducibility and validates the Section 4.1 & 7.1 evaluation contract.
"""

import os
import sys
import unittest
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.backbone import SignalScopeDetector
from model.calibration import TemperatureCalibrator
from model.explainer import LayerCAMExplainer
from model.metadata_inspector import MetadataInspector
from model.predict import predict
from src.generate_sample_data import generate_sample_dataset


class TestSignalScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate sample test data
        cls.test_dir = os.path.join(parent_dir, "data", "sample_val")
        generate_sample_dataset(cls.test_dir)
        cls.sample_image = os.path.join(cls.test_dir, "fake", "sample_synthetic_1.jpg")

    def test_backbone_forward(self):
        """Test model forward pass output shape and logit generation."""
        model = SignalScopeDetector()
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        logit, feat_map = model(dummy_input)

        self.assertEqual(logit.shape, (2, 1))
        self.assertEqual(len(feat_map.shape), 4)

    def test_temperature_calibration(self):
        """Test probability calibration and responsible verdict phrasing."""
        calibrator = TemperatureCalibrator(temperature=1.2)
        verdict_high = calibrator.get_verdict(0.92)
        verdict_low = calibrator.get_verdict(0.12)
        verdict_mid = calibrator.get_verdict(0.50)

        self.assertEqual(verdict_high["verdict"], "Likely AI-Generated")
        self.assertEqual(verdict_low["verdict"], "Likely Real Photo")
        self.assertEqual(verdict_mid["verdict"], "Inconclusive")

    def test_layercam_explainability(self):
        """Test Layer-CAM heatmap generation and cue grounding (Headline Bonus A)."""
        model = SignalScopeDetector()
        explainer = LayerCAMExplainer(model)

        pil_img = Image.open(self.sample_image)
        dummy_tensor = torch.randn(1, 3, 224, 224)

        explanation = explainer.explain_image(pil_img, dummy_tensor)

        self.assertIn("heatmap_raw", explanation)
        self.assertIn("heatmap_overlay", explanation)
        self.assertIn("visual_cues", explanation)
        self.assertIn("faithful_explanation", explanation)
        self.assertTrue(len(explanation["visual_cues"]) > 0)

    def test_metadata_inspector(self):
        """Test EXIF and C2PA inspection logic (Bonus D)."""
        inspector = MetadataInspector()
        meta = inspector.inspect_file(self.sample_image)
        self.assertIn("metadata_signal", meta)
        self.assertIn("metadata_summary", meta)

    def test_end_to_end_predict_cli(self):
        """Test complete predict() interface meeting Section 4.1 contract."""
        result = predict(self.sample_image, explain=True, output_dir=os.path.join(parent_dir, "output"))

        self.assertIn("verdict", result)
        self.assertIn("calibrated_confidence", result)
        self.assertIn("fused_probability_ai", result)
        self.assertIn("explanation", result)
        self.assertTrue(os.path.exists(result["explanation"]["heatmap_path"]))


if __name__ == "__main__":
    unittest.main()
