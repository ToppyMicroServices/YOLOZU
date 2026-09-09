"""Compare product-audit runtime changes against the immutable base revision."""

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', default='b5f5601', help='Immutable Git revision containing the original evaluator.')
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('runtime-benchmark.json'),
                    help='JSON report path (default: beside this audit script).')
args = parser.parse_args()
BASE = args.baseline

from yolozu.eval.simple_map import evaluate_map
from yolozu.datasets.dataset_validator import validate_dataset_records


def baseline(name, path):
    source = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT, text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


old_map = baseline('baseline_simple_map', 'yolozu/eval/simple_map.py').evaluate_map
old_validate = baseline('baseline_dataset_validator', 'yolozu/datasets/dataset_validator.py').validate_dataset_records
thresholds = [0.5 + 0.05 * j for j in range(10)]


def fixture(n_images=2000, classes=20):
    records, predictions = [], []
    for i in range(n_images):
        box = {'cx': 0.5, 'cy': 0.5, 'w': 0.2, 'h': 0.2}
        records.append({'image': f'/data/images/{i}.jpg', 'labels': [{'class_id': i % classes, **box}]})
        predictions.append({'image': f'{i}.jpg', 'detections': [
            {'class_id': i % classes, 'score': 0.9, 'bbox': box},
            {'class_id': i % classes, 'score': 0.2, 'bbox': {'cx': 0.1, 'cy': 0.1, 'w': 0.1, 'h': 0.1}},
        ]})
    return records, predictions


records, predictions = fixture()
timings = {'before': [], 'after': []}
for _ in range(3):
    for label, evaluator in [('before', old_map), ('after', evaluate_map)]:
        start = time.perf_counter()
        result = evaluator(records, predictions, iou_thresholds=thresholds)
        timings[label].append(time.perf_counter() - start)
        assert result.map50 == result.map50_95 == 1.0

stream_measurements = {}
for label, validator in [('before', old_validate), ('after', validate_dataset_records)]:
    tracemalloc.start()
    start = time.perf_counter()
    result = validator(({'image': f'{i}.jpg', 'labels': []} for i in range(100000)), check_images=False)
    duration = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.ok()
    stream_measurements[label] = {'seconds_with_tracemalloc': duration, 'peak_allocated_bytes': peak}

for seed in range(100):
    rng = random.Random(seed)
    records, predictions = [], []
    for idx in range(12):
        labels = []
        for _ in range(rng.randrange(5)):
            labels.append({'class_id': rng.randrange(3), 'cx': rng.uniform(.2, .8),
                           'cy': rng.uniform(.2, .8), 'w': .2, 'h': .2})
        records.append({'image': f'/data/images/{idx}.jpg', 'labels': labels})
        detections = []
        for label in labels:
            if rng.random() > .2:
                bbox = {key: value for key, value in label.items() if key != 'class_id'}
                bbox['cx'] += rng.uniform(-.08, .08)
                detections.append({'class_id': label['class_id'], 'score': rng.choice([.5, .7, .9]), 'bbox': bbox})
        detections.append({'class_id': rng.randrange(4), 'score': .5,
                           'bbox': {'cx': .5, 'cy': .5, 'w': .2, 'h': .2}})
        image = f'{idx}.jpg' if idx % 2 else rf'C:\images\{idx}.jpg'
        predictions.append({'image': image, 'detections': detections})
    predictions.append({'image': 'unknown.jpg', 'detections': [
        {'class_id': 0, 'score': .9, 'bbox': {'cx': .5, 'cy': .5, 'w': .2, 'h': .2}},
    ]})
    for cuts in [[], [0.5], thresholds, [0.75, 0.5]]:
        expected = asdict(old_map(records, predictions, iou_thresholds=cuts))
        actual = asdict(evaluate_map(records, predictions, iou_thresholds=cuts))
        assert expected == actual, (seed, cuts, expected, actual)

output = {
    'baseline_revision': BASE,
    'environment': {'platform': platform.platform(), 'python': sys.version, 'executable_name': Path(sys.executable).name},
    'scope': 'Local synthetic CPU evaluation and dataset-validation overhead; no inference or accuracy claim.',
    'simple_map': {'images': 2000, 'ground_truth_boxes': 2000, 'detections': 4000, 'classes': 20,
                   'thresholds': 10, 'seconds': timings,
                   'median_seconds': {label: statistics.median(values) for label, values in timings.items()},
                   'speedup': statistics.median(timings['before']) / statistics.median(timings['after']),
                   'metrics_before_after': {'map50': 1.0, 'map50_95': 1.0}},
    'dataset_validation': {'generated_records': 100000, 'check_images': False, 'measurements': stream_measurements},
    'randomized_metric_parity': {'seeds': 100, 'threshold_configurations_per_seed': 4, 'exact_equal': True},
}
report = args.output
report.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
