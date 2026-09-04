import csv
import os
import tempfile
import unittest

from ee_utils import (
    _ensure_no_overlap,
    compute_validation_metrics,
    load_ground_truth_csv,
    validate_split_configuration,
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


class ValidationLogicTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
