import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


def load_kn_neurons(kn_neurons_path: Path) -> set[tuple[int, int]]:
    """
    Load the knowledge neurons from a JSON file.

    Args:
        kn_neurons_path (Path): Path to the JSON file containing knowledge neurons.

    """
    try:
        with open(kn_neurons_path, "r") as f:
            kn_neurons = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {kn_neurons_path}")
        return None


    neurons_set = set()
    for elem in kn_neurons:
        if isinstance(elem, list) and len(elem) == 2:
            layer, neuron = elem
            neurons_set.add((layer, neuron))
    return neurons_set


def create_kn_neurons_union(kn_neurons_paths: list) -> dict:
    """
    Create a union mask of knowledge neurons from multiple JSON files.

    Args:
        kn_neurons_paths (list): List of paths to JSON files containing knowledge neurons.
    """
    kn_neurons_union = defaultdict(set)

    for path in kn_neurons_paths:
        kn_neurons = load_kn_neurons(path)

        for layer, neuron in kn_neurons:
            kn_neurons_union[layer].add(neuron)

    return dict(kn_neurons_union)


def create_kn_neurons_intersection(
    kn_neurons_paths: list[Path]
) -> dict:
    """
    Compute neurons shared by ALL supplied kn_rel files.
    """

    if not kn_neurons_paths:
        return dict()

    kn_neurons_intersection = defaultdict(set)
    for path in kn_neurons_paths:
        kn_neurons = load_kn_neurons(path)
    
        for layer, neuron in kn_neurons:
            kn_neurons_intersection[layer].add(neuron)
    
    return dict(kn_neurons_intersection)




def create_mask_foundation(kn_neurons_directory: Path, is_union: bool = True, is_intersection: bool = False) -> dict:
    """
    Create a mask of knowledge neurons based on the specified directory and options.
    """

    kn_neurons_paths = list(kn_neurons_directory.glob("kn_rel*.json"))
    print(kn_neurons_paths)
    kn_neurons_union = {}
    kn_neurons_intersection = {}

    kn_neurons_mask_generator = {}

    if is_union:
        kn_neurons_union = create_kn_neurons_union(kn_neurons_paths)
    
    if is_intersection:
        kn_neurons_intersection = create_kn_neurons_intersection(kn_neurons_paths)

    kn_neurons_mask_generator = {
        "union": kn_neurons_union,
        "intersection": kn_neurons_intersection,
    }

    return kn_neurons_mask_generator


