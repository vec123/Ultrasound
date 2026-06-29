import json
import numpy as np
def load_landmarks(json_path):
  

    with open(json_path, "r") as f:
        data = json.load(f)

    landmarks = {}
    names = []
    positions = []

    # Case 1: dict format
    if isinstance(data, dict):
        for k, v in data.items():
            landmarks[k] = np.asarray(v, dtype=np.float32)
            names.append(k)
            positions.append(v)

    # Case 2: list format
    elif isinstance(data, list):
        for item in data:
            name = item.get("name")
            pos = item.get("position")

            landmarks[name] = np.asarray(pos, dtype=np.float32)
            names.append(name)
            positions.append(pos)

    else:
        raise ValueError("Unsupported landmark JSON format")

    positions = np.asarray(positions, dtype=np.float32)

    return landmarks, positions, names