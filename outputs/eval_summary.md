# Eval summary

n = 17 (golden set)

| | Intent accuracy | Decision accuracy |
|---|---|---|
| System | 0.82 | 0.88 |
| Trivial baseline (majority class) | 0.29 | 0.59 |
| Simple keyword baseline | 0.41 | 0.65 |

## Per-class intent report (system)

| intent | support | precision | recall | f1 |
|---|---|---|---|---|
| account_appstore | 1 | 1.0 | 1.0 | 1.0 |
| app_crash_freeze | 1 | 1.0 | 1.0 | 1.0 |
| battery_drain | 5 | 1.0 | 0.6 | 0.75 |
| connectivity_issue | 1 | 1.0 | 1.0 | 1.0 |
| feature_conflict | 1 | 1.0 | 1.0 | 1.0 |
| general_update_complaint | 2 | 1.0 | 1.0 | 1.0 |
| other | 3 | 0.5 | 1.0 | 0.67 |
| performance_lag | 2 | 1.0 | 1.0 | 1.0 |
| praise_feedback | 1 | 0.0 | 0.0 | 0.0 |

## Intent confusion (gold -> predicted counts)

- **other**: {'other': 3}
- **performance_lag**: {'performance_lag': 2}
- **connectivity_issue**: {'connectivity_issue': 1}
- **battery_drain**: {'other': 2, 'battery_drain': 3}
- **general_update_complaint**: {'general_update_complaint': 2}
- **feature_conflict**: {'feature_conflict': 1}
- **account_appstore**: {'account_appstore': 1}
- **app_crash_freeze**: {'app_crash_freeze': 1}
- **praise_feedback**: {'other': 1}

## Decision confusion (gold -> predicted counts)

- **escalate**: {'escalate': 7}
- **auto_handle**: {'escalate': 2, 'auto_handle': 8}

## Reply quality (judge: heuristic)

Average composite score: 0.96 / 1.00 (4-check rubric)

### Agreement with human-scored labels

n = 17

| check | agreement |
|---|---|
| grounded | 1.00 |
| on_topic | 0.82 |
| not_redundant | 0.82 |
| tone_appropriate | 0.94 |
