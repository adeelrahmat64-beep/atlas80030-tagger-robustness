"""Real-cache smoke coverage for the modelling inference schemas.

Run with:
    python -m unittest "Code files/test_atlas_tagger_evaluation_schema.py"
"""

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import atlas_tagger
from atlas_config import MODELLING_DATASETS


class ModellingInferenceSchemaTest(unittest.TestCase):
    def test_all_six_real_modelling_schemas_load_without_event_numbers(self):
        """Inference accepts every real modelling cache without fabricating IDs."""
        for dataset in MODELLING_DATASETS:
            with self.subTest(dataset=dataset):
                X, labels, pt, event_number, _sizes, keep, _stems = (
                    atlas_tagger.load_features(
                        dataset, limit=1, require_event_number=False
                    )
                )
                self.assertEqual(X.shape, (1, len(atlas_tagger.HL)))
                self.assertEqual(labels.shape, (1,))
                self.assertEqual(pt.shape, (1,))
                self.assertEqual(keep.shape, (1,))
                self.assertIsNone(event_number)


class FinalPredictionManifestTest(unittest.TestCase):
    def test_manifest_identity_rejects_a_different_final_model(self):
        metadata = {
            "model_sha256": "model-hash",
            "tagger_source_sha256": "source-hash",
            "reconstruction_fingerprint": atlas_tagger.FROZEN_RECONSTRUCTION_FP,
            "training_rows_requested": 5_000_000,
            "seed": 42,
        }
        manifest = atlas_tagger.final_prediction_manifest(
            Path("atlas_tagger_xgb_final_production_round1.joblib"), metadata
        )
        self.assertTrue(atlas_tagger.manifest_matches_final_model(manifest, manifest))
        different_model = dict(manifest)
        different_model["model_sha256"] = "other-model-hash"
        self.assertFalse(
            atlas_tagger.manifest_matches_final_model(different_model, manifest)
        )
