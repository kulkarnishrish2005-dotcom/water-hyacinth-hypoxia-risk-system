import csv
import os
import tempfile
import unittest

from ee_utils import (
    _ensure_no_overlap,
    compute_validation_metrics,
    load_ground_truth_csv,
    validate_split_configuration,
    normalize_class_label,
    PHASE2_CONFIG,
)


class FakeGetInfo:
    def __init__(self, value):
        self._value = value

    def getInfo(self):
        return self._value


class FakeEECollection:
    def __init__(self, values=None, class_values=None):
        self._values = values or []
        self._class_values = class_values or []

    def size(self):
        return FakeGetInfo(len(self._values))

    def aggregate_array(self, field):
        if field == 'class':
            return FakeGetInfo(self._class_values)
        if field == 'site':
            return FakeGetInfo(sorted({item['site'] for item in self._values}))
        if field == 'region':
            return FakeGetInfo(sorted({item['region'] for item in self._values}))
        if field == 'sector':
            return FakeGetInfo(sorted({item['sector'] for item in self._values}))
        if field == 'sample_id':
            return FakeGetInfo([item['sample_id'] for item in self._values])
        return FakeGetInfo([])

    def aggregate_histogram(self, field):
        if field == 'class':
            counts = {}
            for value in self._class_values:
                counts[value] = counts.get(value, 0) + 1
            return FakeGetInfo(counts)
        return FakeGetInfo({})


class Phase1ValidationTests(unittest.TestCase):
    def test_invalid_class_rejected(self):
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'site', 'region', 'country', 'class', 'label', 'lat', 'lon', 'date', 'source', 'sample_type', 'sector', 'notes'
            ])
            writer.writerow(['site_a', 'north', 'India', '99', 'Bad Class', '12.5', '77.5', '2025-02-01', 'manual', 'point', 'north', 'bad'])
            path = csv_file.name

        try:
            with self.assertRaises(ValueError):
                load_ground_truth_csv(path)
        finally:
            os.unlink(path)

    def test_duplicate_coordinates_rejected(self):
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'site', 'region', 'country', 'class', 'label', 'lat', 'lon', 'date', 'source', 'sample_type', 'sector', 'notes'
            ])
            writer.writerow(['site_a', 'north', 'India', '0', 'Water', '12.5', '77.5', '2025-02-01', 'manual', 'point', 'north', 'one'])
            writer.writerow(['site_a', 'north', 'India', '1', 'Water Hyacinth', '12.5', '77.5', '2025-02-01', 'manual', 'point', 'north', 'two'])
            path = csv_file.name

        try:
            with self.assertRaises(ValueError):
                load_ground_truth_csv(path)
        finally:
            os.unlink(path)

    def test_overlap_train_validation_sectors(self):
        with self.assertRaises(ValueError):
            _ensure_no_overlap(['north'], ['north'], 'sectors')

    def test_overlap_train_validation_sites(self):
        with self.assertRaises(ValueError):
            _ensure_no_overlap(['site_a'], ['site_a'], 'sites')

    def test_missing_validation_data(self):
        train_fc = FakeEECollection(
            values=[{'site': 's1', 'region': 'r1', 'sector': 'n1', 'sample_id': 'a'}],
            class_values=[0, 1, 2]
        )
        validation_fc = FakeEECollection(values=[], class_values=[])
        with self.assertRaises(ValueError):
            validate_split_configuration(train_fc, validation_fc)

    def test_missing_class_in_training(self):
        train_fc = FakeEECollection(
            values=[{'site': 's1', 'region': 'r1', 'sector': 'n1', 'sample_id': 'a'}],
            class_values=[0, 1]
        )
        validation_fc = FakeEECollection(
            values=[{'site': 's2', 'region': 'r2', 'sector': 'n2', 'sample_id': 'b'}],
            class_values=[0, 1, 2]
        )
        with self.assertRaises(ValueError):
            validate_split_configuration(train_fc, validation_fc)

    def test_same_patch_in_train_and_validation(self):
        with self.assertRaises(ValueError):
            _ensure_no_overlap(['patch-1'], ['patch-1'], 'patches')

    def test_metric_calculation(self):
        confusion = [[10, 1], [2, 20]]
        metrics = compute_validation_metrics(confusion, class_labels=['Water', 'Water Hyacinth'])
        self.assertAlmostEqual(metrics['overall_accuracy'], 30 / 33, places=6)
        self.assertAlmostEqual(metrics['per_class']['Water']['precision'], 10 / 12, places=6)
        self.assertAlmostEqual(metrics['per_class']['Water']['recall'], 10 / 11, places=6)
        self.assertAlmostEqual(metrics['macro_f1'], (metrics['per_class']['Water']['f1'] + metrics['per_class']['Water Hyacinth']['f1']) / 2, places=6)

    def test_valid_split_passes(self):
        train_fc = FakeEECollection(
            values=[
                {'site': 'site_a', 'region': 'north', 'sector': 'north', 'sample_id': 'a'},
                {'site': 'site_a', 'region': 'north', 'sector': 'north', 'sample_id': 'b'},
                {'site': 'site_a', 'region': 'north', 'sector': 'north', 'sample_id': 'c'},
            ],
            class_values=[0, 1, 2],
        )
        validation_fc = FakeEECollection(
            values=[
                {'site': 'site_b', 'region': 'south', 'sector': 'south', 'sample_id': 'd'},
                {'site': 'site_b', 'region': 'south', 'sector': 'south', 'sample_id': 'e'},
                {'site': 'site_b', 'region': 'south', 'sector': 'south', 'sample_id': 'f'},
            ],
            class_values=[0, 1, 2],
        )
        result = validate_split_configuration(train_fc, validation_fc)
        self.assertEqual(result['train_size'], 3)
        self.assertEqual(result['validation_size'], 3)


class Phase2Tests(unittest.TestCase):
    def test_class_normalization(self):
        self.assertEqual(normalize_class_label(0), 0)
        self.assertEqual(normalize_class_label('0'), 0)
        self.assertEqual(normalize_class_label('water'), 0)
        self.assertEqual(normalize_class_label(1), 1)
        self.assertEqual(normalize_class_label('water hyacinth'), 1)
        self.assertEqual(normalize_class_label('hyacinth'), 1)
        self.assertEqual(normalize_class_label(2), 2)
        self.assertEqual(normalize_class_label('other vegetation'), 2)
        self.assertEqual(normalize_class_label('lotus'), 2)
        self.assertEqual(normalize_class_label('lotus-like'), 2)

    def test_invalid_class_normalization(self):
        with self.assertRaises(ValueError):
            normalize_class_label(99)
        with self.assertRaises(ValueError):
            normalize_class_label('unknown_class')

    def test_phase2_config_structure(self):
        self.assertIn('chlorophyll_proxy', PHASE2_CONFIG)
        self.assertIn('turbidity_proxy', PHASE2_CONFIG)
        self.assertIn('temperature', PHASE2_CONFIG)
        self.assertIn('hypoxia_risk', PHASE2_CONFIG)
        self.assertIn('hhri', PHASE2_CONFIG)

        # Check HHRI default weights
        hhri = PHASE2_CONFIG['hhri']
        self.assertIn('weights', hhri)
        self.assertIn('thresholds', hhri)
        self.assertEqual(hhri['weights']['w1_ndvi'], 1.0)
        self.assertEqual(hhri['weights']['w2_ndwi'], 1.0)
        self.assertEqual(hhri['weights']['w3_chl'], 1.0)
        self.assertEqual(hhri['weights']['w4_turbidity'], 1.0)
        self.assertEqual(hhri['weights']['w5_doproxy'], 1.0)

        # Check HHRI default thresholds
        self.assertEqual(hhri['thresholds']['low'], 0.3)
        self.assertEqual(hhri['thresholds']['moderate'], 0.6)

    def test_phase2_chlorophyll_config(self):
        chl = PHASE2_CONFIG['chlorophyll_proxy']
        self.assertEqual(chl['formula'], 'Gitelson')
        self.assertEqual(chl['red_edge_band'], 'B5')
        self.assertEqual(chl['red_band'], 'B4')
        self.assertEqual(chl['nir_band'], 'B8')
        self.assertIn('coefficients', chl)


if __name__ == '__main__':
    unittest.main()


class Phase2NumericTypeRegression(unittest.TestCase):
    """Regression: EE Image must call .multiply(), never Python float .multiply()."""

    def test_chlorophyll_ee_image_calls_multiply_not_float(self):
        """Line 1105 fix: coefficient applied by EE Image, not float scalar."""
        from ee_utils import compute_chlorophyll_proxy, PHASE2_CONFIG
        # If coefficients['L'] (2.56 float) called .multiply(), AttributeError.
        # After fix: ratio (ee.Image) calls .multiply(2.56) — valid.
        coeffs = PHASE2_CONFIG['chlorophyll_proxy']['coefficients']
        self.assertIsInstance(coeffs['L'], float)  # still Python float in config
        # The formula structure is preserved; only method caller changed.
        # We verify by inspecting the source line pattern isn't float.multiply.
        import inspect
        src = inspect.getsource(compute_chlorophyll_proxy)
        self.assertNotIn("coefficients['L'].multiply(", src,
                         "Bug: float scalar calling .multiply()")
        self.assertIn(".multiply(coefficients['L'])", src,
                      "Fix: EE Image calling multiply with coeff")
