import os
import json
from configs import PREDICTION_DIR


def write_predictions(predicted_strings, file_path):
    write_file = os.path.join(PREDICTION_DIR, file_path)
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    with open(write_file, 'w+', encoding='utf-8') as f:
        for p in predicted_strings:
            f.write(json.dumps(p)+"\n")


def write_reference(predicted_strings, file_path):
    write_file = os.path.join(PREDICTION_DIR, file_path)
    os.makedirs(os.path.dirname(write_file), exist_ok=True)
    with open(write_file, 'w+', encoding='utf-8') as f:
        for p in predicted_strings:
            f.write('{}\n'.format(p))
