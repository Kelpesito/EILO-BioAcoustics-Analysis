"""
class_to_idx.py

Dictionaries to relate the original label to ordinal encoding label
"""


MULTICLASS_IDX = {
    "Normal": 0,
    "Fine Crackle": 1,
    "Coarse Crackle": 2,
    "Wheeze": 3,
    "Wheeze+Crackle": 4,
    "Rhonchi": 5,
    "Stridor": 6,
}

FILTERED_IDX = {
    "Normal": 0,
    "Wheeze": 1,
    "Rhonchi": 2,
    "Stridor": 3,
}

BINARY_IDX = {
    "Normal": 0,
    "Adventitious": 1,
}
