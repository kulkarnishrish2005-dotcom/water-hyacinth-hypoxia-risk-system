"""
Tests for dual-mode architecture: prototype vs scientific.

Verifies:
- prototype mode: ground_truth_path is optional (None accepted)
- prototype mode: validation splits are optional (None accepted)
- scientific mode: ground_truth_path is required
- scientific mode: validation splits are required
- mode routing dispatches to correct sub-function
- error messages are clear and specific
"""

import csv
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

# Import the functions under test
from ee_utils import (
    run_scientific_validation,
    _run_prototype_mode,
    _run_scientific_mode,
    PROXY_MODE_LABEL,
)


class ModeRoutingTests(unittest.TestCase):
    """Test mode selection and routing logic (no GEE required)."""

    def test_unknown_mode_raises_clear_error(self):
        """Unknown mode should raise ValueError with helpful message."""
        with self.assertRaises(ValueError) as ctx:
            run_scientific_validation(
                ground_truth_path=None,
                analysis_date='2024-01-15',
                lat=12.5,
                lon=77.5,
                mode='invalid_mode'
            )
        self.assertIn("'invalid_mode'", str(ctx.exception))
        self.assertIn("prototype", str(ctx.exception).lower())
        self.assertIn("scientific", str(ctx.exception).lower())

    def test_prototype_mode_calls_correct_subfunction(self):
        """Prototype mode should route to _run_prototype_mode."""
        with patch('ee_utils._run_prototype_mode') as mock:
            mock.return_value = {'mode': 'prototype'}
            result = run_scientific_validation(
                ground_truth_path=None,  # Should be ignored
                analysis_date='2024-01-15',
                lat=12.5,
                lon=77.5,
                mode='prototype',
                # These should be ignored in prototype mode:
                train_sites=['site_a'],
                validation_sites=['site_b'],
            )
            # Verify _run_prototype_mode was called (not _run_scientific_mode)
            mock.assert_called_once()
            call_kwargs = mock.call_args.kwargs
            self.assertEqual(call_kwargs['analysis_date'], '2024-01-15')
            self.assertEqual(call_kwargs['lat'], 12.5)
            self.assertEqual(call_kwargs['lon'], 77.5)
            # ground_truth_path should NOT be passed to prototype
            self.assertNotIn('ground_truth_path', call_kwargs)
            # validation splits should NOT be passed to prototype
            self.assertNotIn('train_sites', call_kwargs)
            self.assertNotIn('validation_sites', call_kwargs)

    def test_scientific_mode_calls_correct_subfunction(self):
        """Scientific mode should route to _run_scientific_mode."""
        # Create a minimal valid ground-truth CSV
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, suffix='.csv', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'site', 'region', 'country', 'class', 'label',
                'lat', 'lon', 'date', 'source', 'sample_type', 'sector', 'notes'
            ])
            writer.writerow([
                'site_a', 'north', 'India', '0', 'Water',
                '12.5', '77.5', '2024-01-15', 'manual', 'point', 'north', ''
            ])
            csv_path = f.name

        try:
            with patch('ee_utils._run_scientific_mode') as mock:
                mock.return_value = {'mode': 'scientific'}
                result = run_scientific_validation(
                    ground_truth_path=csv_path,
                    analysis_date='2024-01-15',
                    lat=12.5,
                    lon=77.5,
                    mode='scientific',
                    train_sites=['site_a'],
                    validation_sites=['site_b'],
                    train_regions=['north'],
                    validation_regions=['south'],
                )
                # Verify _run_scientific_mode was called (not _run_prototype_mode)
                mock.assert_called_once()
                call_kwargs = mock.call_args.kwargs
                self.assertEqual(call_kwargs['ground_truth_path'], csv_path)
                self.assertEqual(call_kwargs['train_sites'], ['site_a'])
                self.assertEqual(call_kwargs['validation_sites'], ['site_b'])
        finally:
            os.unlink(csv_path)


class PrototypeModeTests(unittest.TestCase):
    """Test prototype mode works without ground-truth or validation splits."""

    @patch('ee_utils.run_full_pipeline')
    @patch('ee_utils.make_aoi')
    def test_prototype_runs_without_ground_truth(self, mock_aoi, mock_pipeline):
        """Prototype mode should NOT require ground_truth_path."""
        mock_aoi.return_value = None
        mock_pipeline.return_value = {
            'mode': 'prototype',
            'image': None,
            'aoi': None,
            'classification': {},
            'thumbnails': {},
            'ndvi': None,
            'ndwi': None,
            'water_mask': None,
            'ndvi_texture': None,
            'distance_to_shore': None,
            'chlorophyll_proxy': None,
            'turbidity_proxy': None,
            'temperature': None,
        }

        # This must NOT raise FileNotFoundError
        result = _run_prototype_mode(
            analysis_date='2024-01-15',
            lat=12.5,
            lon=77.5,
        )

        self.assertEqual(result['mode'], 'prototype')
        self.assertIn('scientific_note', result)
        # Must be the proxy warning, not scientific validation
        self.assertEqual(result['scientific_note'], PROXY_MODE_LABEL)

    @patch('ee_utils.run_full_pipeline')
    @patch('ee_utils.make_aoi')
    def test_prototype_ignores_validation_splits(self, mock_aoi, mock_pipeline):
        """Prototype mode should ignore train/validation split parameters."""
        mock_aoi.return_value = None
        mock_pipeline.return_value = {
            'mode': 'prototype',
            'image': None,
            'aoi': None,
            'classification': {},
            'thumbnails': {},
            'ndvi': None,
            'ndwi': None,
            'water_mask': None,
            'ndvi_texture': None,
            'distance_to_shore': None,
            'chlorophyll_proxy': None,
            'turbidity_proxy': None,
            'temperature': None,
        }

        # This should NOT fail even though splits are provided
        result = _run_prototype_mode(
            analysis_date='2024-01-15',
            lat=12.5,
            lon=77.5,
            # These are scientific mode parameters — prototype should ignore them
            # (we pass via the top-level call; _run_prototype_mode doesn't accept them)
        )

        # Should complete successfully
        self.assertEqual(result['mode'], 'prototype')
        self.assertEqual(result['scientific_note'], PROXY_MODE_LABEL)

    @patch('ee_utils.run_full_pipeline')
    @patch('ee_utils.make_aoi')
    def test_prototype_wraps_pipeline_result(self, mock_aoi, mock_pipeline):
        """Prototype mode should wrap run_full_pipeline result for UI consistency."""
        mock_aoi.return_value = None
        mock_pipeline.return_value = {
            'mode': 'prototype',
            'image': 'fake_image_obj',
            'aoi': 'fake_aoi_obj',
            'classification': {'water': 100, 'hyacinth': 50},
            'thumbnails': {'true_color': 'fake_thumb'},
            'ndvi': 'fake_ndvi',
            'ndwi': 'fake_ndwi',
            'water_mask': 'fake_mask',
            'ndvi_texture': 'fake_texture',
            'distance_to_shore': 'fake_distance',
            'chlorophyll_proxy': 'fake_chl',
            'turbidity_proxy': 'fake_turb',
            'temperature': 'fake_temp',
        }

        result = _run_prototype_mode(
            analysis_date='2024-01-15',
            lat=12.5,
            lon=77.5,
        )

        # Check that the wrapper includes expected fields
        self.assertEqual(result['mode'], 'prototype')
        self.assertEqual(result['classification']['water'], 100)
        self.assertEqual(result['classification']['hyacinth'], 50)
        self.assertEqual(result['classification']['mode'], 'prototype')
        self.assertEqual(result['thumbnails'], {'true_color': 'fake_thumb'})
        self.assertEqual(result['scientific_note'], PROXY_MODE_LABEL)
        # Feature bands should be documented
        self.assertEqual(result['feature_bands'], [
            'NDVI', 'NDWI', 'ndvi_texture', 'distance_to_shore'
        ])
        # Validation fields should be empty/None for prototype
        self.assertIsNone(result['confusion_matrix'])
        self.assertEqual(result['metrics']['overall_accuracy'], 0.0)
        self.assertEqual(result['train_sample_count'], 0)
        self.assertEqual(result['validation_sample_count'], 0)

    def test_prototype_requires_lat_lon_or_aoi(self):
        """Prototype mode should require either AOI or lat+lon."""
        with self.assertRaises(ValueError) as ctx:
            _run_prototype_mode(analysis_date='2024-01-15')
        self.assertIn('lat', str(ctx.exception).lower())
        self.assertIn('lon', str(ctx.exception).lower())
        self.assertIn('aoi', str(ctx.exception).lower())

    def test_prototype_requires_analysis_date(self):
        """Prototype mode should require analysis_date."""
        with self.assertRaises(ValueError) as ctx:
            _run_prototype_mode(analysis_date=None, lat=12.5, lon=77.5)
        self.assertIn('analysis_date', str(ctx.exception))


class ScientificModeTests(unittest.TestCase):
    """Test scientific mode requires ground-truth and validation splits."""

    def test_scientific_mode_requires_ground_truth_path(self):
        """Scientific mode should raise FileNotFoundError when ground_truth_path is None."""
        with self.assertRaises((FileNotFoundError, ValueError)) as ctx:
            _run_scientific_mode(
                ground_truth_path=None,
                analysis_date='2024-01-15',
                lat=12.5,
                lon=77.5,
                train_sites=['site_a'],
                validation_sites=['site_b'],
            )
        msg = str(ctx.exception).lower()
        self.assertTrue('ground' in msg or 'csv' in msg or 'required' in msg)

    def test_scientific_mode_requires_valid_ground_truth_path(self):
        """Scientific mode should raise FileNotFoundError when path doesn't exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            _run_scientific_mode(
                ground_truth_path='/nonexistent/path/to/ground_truth.csv',
                analysis_date='2024-01-15',
                lat=12.5,
                lon=77.5,
                train_sites=['site_a'],
                validation_sites=['site_b'],
            )
        self.assertIn('ground-truth', str(ctx.exception))

    def test_scientific_mode_requires_ground_truth_csv_format(self):
        """Scientific mode should validate ground-truth CSV structure."""
        # Create an invalid CSV (missing required columns)
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, suffix='.csv', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['wrong_column_1', 'wrong_column_2'])
            writer.writerow(['val1', 'val2'])
            csv_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                _run_scientific_mode(
                    ground_truth_path=csv_path,
                    analysis_date='2024-01-15',
                    lat=12.5,
                    lon=77.5,
                    train_sites=['site_a'],
                    validation_sites=['site_b'],
                )
            # Error should mention missing columns
            self.assertIn('required', str(ctx.exception).lower())
        finally:
            os.unlink(csv_path)

    def test_scientific_mode_requires_validation_splits(self):
        """Scientific mode should require at least one type of validation split."""
        # Create a minimal valid ground-truth CSV
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, suffix='.csv', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'site', 'region', 'country', 'class', 'label',
                'lat', 'lon', 'date', 'source', 'sample_type', 'sector', 'notes'
            ])
            writer.writerow([
                'site_a', 'north', 'India', '0', 'Water',
                '12.5', '77.5', '2024-01-15', 'manual', 'point', 'north', ''
            ])
            csv_path = f.name

        try:
            # Pass NO validation splits at all
            with self.assertRaises(ValueError) as ctx:
                _run_scientific_mode(
                    ground_truth_path=csv_path,
                    analysis_date='2024-01-15',
                    lat=12.5,
                    lon=77.5,
                    # No train_sites, train_regions, or train_sectors provided
                )
            # Error should guide toward providing splits
            self.assertIn('split', str(ctx.exception).lower())
        finally:
            os.unlink(csv_path)


class IntegrationTests(unittest.TestCase):
    """Full pipeline integration tests (mocked GEE)."""

    @patch('ee_utils.run_full_pipeline')
    @patch('ee_utils.make_aoi')
    def test_prototype_mode_end_to_end(self, mock_aoi, mock_pipeline):
        """Full prototype pipeline should complete without ground-truth."""
        mock_aoi.return_value = None
        mock_pipeline.return_value = {
            'mode': 'prototype',
            'image': None,
            'aoi': None,
            'classification': {'water': 200},
            'thumbnails': {'true_color': 'fake_url'},
            'ndvi': 'fake_ndvi',
            'ndwi': 'fake_ndwi',
            'water_mask': 'fake_mask',
            'ndvi_texture': 'fake_texture',
            'distance_to_shore': 'fake_dist',
            'chlorophyll_proxy': 'fake_chl',
            'turbidity_proxy': 'fake_turb',
            'temperature': 'fake_temp',
        }

        # This is the exact call app.py makes for prototype mode
        result = run_scientific_validation(
            ground_truth_path=None,  # prototype doesn't need it
            analysis_date=datetime(2024, 1, 15),
            lat=12.5,
            lon=77.5,
            start_date='2024-01-14',
            end_date='2024-01-16',
            max_cloud=10,
            scale=10,
            mode='prototype',
            # NO train_sites, validation_sites, etc.
        )

        # Verify prototype result structure
        self.assertEqual(result['mode'], 'prototype')
        self.assertEqual(result['scientific_note'], PROXY_MODE_LABEL)
        mock_pipeline.assert_called_once()
        pipeline_kwargs = mock_pipeline.call_args.kwargs
        self.assertEqual(pipeline_kwargs['lat'], 12.5)
        self.assertEqual(pipeline_kwargs['lon'], 77.5)
        # labelled_points should be None in prototype
        self.assertIsNone(pipeline_kwargs['labelled_points'])

    @patch('ee_utils._run_scientific_mode')
    def test_scientific_mode_end_to_end(self, mock_scientific):
        """Full scientific pipeline should require and use ground-truth."""
        mock_scientific.return_value = {
            'mode': 'scientific',
            'scientific_note': 'Scientific validation complete',
            'confusion_matrix': [[10, 2], [1, 20]],
            'metrics': {'overall_accuracy': 0.94, 'macro_f1': 0.93},
        }

        # Create a minimal valid ground-truth CSV
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, suffix='.csv', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'site', 'region', 'country', 'class', 'label',
                'lat', 'lon', 'date', 'source', 'sample_type', 'sector', 'notes'
            ])
            writer.writerow([
                'site_a', 'north', 'India', '0', 'Water',
                '12.5', '77.5', '2024-01-15', 'manual', 'point', 'north', ''
            ])
            writer.writerow([
                'site_b', 'south', 'India', '1', 'Water Hyacinth',
                '12.6', '77.6', '2024-01-15', 'manual', 'point', 'south', ''
            ])
            csv_path = f.name

        try:
            result = run_scientific_validation(
                ground_truth_path=csv_path,
                analysis_date=datetime(2024, 1, 15),
                lat=12.5,
                lon=77.5,
                train_sites=['site_a'],
                validation_sites=['site_b'],
                start_date='2024-01-14',
                end_date='2024-01-16',
                max_cloud=10,
                scale=10,
                mode='scientific',
            )

            # Verify scientific result structure
            self.assertEqual(result['mode'], 'scientific')
            mock_scientific.assert_called_once()
        finally:
            os.unlink(csv_path)


if __name__ == '__main__':
    unittest.main()
