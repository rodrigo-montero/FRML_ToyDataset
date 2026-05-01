# Shortcut Learning Toy Dataset

## Hypothesis
This dataset tests whether a model relies on colour (shortcut feature) instead of shape (intended feature).

## Dataset Design
- Images: 32x32 synthetic shapes
- Labels:
  - 0 = square
  - 1 = circle

## Shortcut Setup
Training:
- square → blue
- circle → red

Test (OOD):
- square → red
- circle → blue

## Generation
Run:
```python generate_dataset.py```

Files 
- 
- data/
  - train/
  - test_id/
  - test_ood/
  - metadata.csv
- generate_dataset.py
- README.md
- run_experiment.py

## Experiment

run_experiment.py runs a small experiment to test the hypothesis. 
The results can be summarised as:
Accuracy on 'Train': 1.0
Accuracy on 'Test ID': 1.0
Accuracy on 'Test OOD': 0.0

