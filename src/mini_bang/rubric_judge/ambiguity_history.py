"""Retain uncertainty from every completed encoding without changing final labels."""
from copy import deepcopy


def ambiguity_history(encodings, *, kind):
    field = 'decisions' if kind == 'controls' else 'coverage'
    observations = []
    for encoding_index, encoding in enumerate(encodings):
        if not isinstance(encoding, dict): continue
        rows = [encoding] if kind == 'coverage' else encoding.get(field, [])
        if not isinstance(rows, list): continue
        for row_index, row in enumerate(rows):
            if isinstance(row, dict) and row.get('scope_ambiguity') is True:
                observations.append({'completed_encoding_index': encoding_index,
                    'row_index': row_index, 'original_decision': deepcopy(row)})
    return {'policy': 'sticky_scientific_hold_across_completed_encodings_v1',
        'scientific_grade_held': bool(observations), 'completed_encodings_inspected': len(encodings),
        'criterion_ids': sorted({str(row['original_decision'].get('id', 'unidentified')) for row in observations}),
        'observations': observations, 'final_labels_changed': False}


def apply_hold(metrics, history):
    """Preserve original metrics/labels/weights; mark a diagnostic value as held."""
    result = deepcopy(metrics)
    result['scope_ambiguity_history'] = deepcopy(history)
    result['scientific_grade_held'] = bool(result.get('scientific_grade_held') or history['scientific_grade_held'])
    result['score_is_diagnostic'] = bool(result.get('score_is_diagnostic') or result['scientific_grade_held'])
    return result
