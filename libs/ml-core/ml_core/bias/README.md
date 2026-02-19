# Bias Detection and Mitigation

This module provides tools for detecting and mitigating bias in ML model predictions through data slicing and analysis.

## Overview

Bias detection ensures the ticket assignment model performs fairly across different subgroups such as repositories, seniority levels, and ticket types. This prevents the model from systematically favoring or disadvantaging certain groups.

## Components

### DataSlicer (slicer.py)

Splits data into subgroups for analysis. Supports slicing by repository, seniority level, labels, completion time buckets, and technical keywords.

### BiasAnalyzer (analyzer.py)

Detects bias by comparing model performance across slices using regression metrics. Calculates MAE, RMSE, and R-squared for each slice and flags significant performance gaps exceeding the threshold.

### BiasMitigator (mitigation.py)

Provides techniques to address detected bias including resampling underrepresented groups to balance dataset, computing sample weights for fair training, and applying fairness threshold adjustments to predictions.

### BiasReport (report.py)

Generates formatted reports summarizing bias analysis results including which dimensions show bias and performance metrics per slice.

## Bias Analysis Results

Analysis of 61,271 tickets revealed the following distribution imbalances. Repository distribution shows Ansible with 33,286 tickets, Terraform with 21,611 tickets, and Prometheus with 6,374 tickets, representing a 5x imbalance between largest and smallest repos. All tickets are currently labeled as mid seniority, indicating the seniority mapping logic needs improvement. Completion time analysis shows 36,088 slow tickets over 24 hours, 14,457 fast tickets under 5 hours, and 7,874 medium tickets, with 2,852 having unknown completion times.

## Mitigation Applied

To address repository imbalance, we implemented resampling to balance the dataset. The resampling process duplicates tickets from underrepresented repos (Terraform and Prometheus) to match the size of the largest repo (Ansible), resulting in 33,286 tickets per repo and a balanced dataset of 99,858 total tickets. We also computed sample weights inversely proportional to group frequency, assigning Prometheus tickets a weight of 3.2042, Terraform tickets 0.9451, and Ansible tickets 0.6136, ensuring underrepresented groups receive higher importance during training.

## Trade-offs

Resampling increases dataset size through duplication but does not add new information, only emphasizes underrepresented patterns. The balanced dataset may overfit to duplicated patterns from small groups. Sample weighting preserves original data without duplication but requires the training algorithm to support weighted samples. Both approaches prioritize fairness across groups while accepting potential minor overall performance reduction.

## Usage

Data slicing example:
```python
from ml_core.bias import DataSlicer
slicer = DataSlicer(data)
slices = slicer.slice_by_repo()
```

Bias detection example:
```python
from ml_core.bias import BiasAnalyzer
analyzer = BiasAnalyzer(threshold=0.1)
result = analyzer.compare_slices(slices, "y_true", "y_pred")
```

Bias mitigation example:
```python
from ml_core.bias import BiasMitigator
mitigator = BiasMitigator()
balanced_data = mitigator.resample_underrepresented(data, "repo")
weights = mitigator.compute_sample_weights(data, "repo")
```

Generate report example:
```python
from ml_core.bias import BiasReport
report = BiasReport.generate_text_report(analysis)
BiasReport.save_report(analysis, "bias_report.txt")
```