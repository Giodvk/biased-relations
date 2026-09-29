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


def create_mask_relation_based(kn_neurons_directory: Path, relation_ids) -> dict:
    kn_neurons_dict = defaultdict(dict)

    for relation_id in relation_ids:
        filename = (kn_neurons_directory/f"kn_rel-{relation_id}.json")
        layer_map = defaultdict(set)
        neuron_set = load_kn_neurons(filename)

        if not filename.exists():
            raise FileNotFoundError(
        f"KN file not found for relation {relation_id}"
        )

        for layer, neuron in neuron_set:
            layer_map[layer].add(neuron)
        kn_neurons_dict[relation_id] = layer_map
    return dict(kn_neurons_dict)
        
        


if __name__ == "__main__":
    relation_ids = ["BR0" + str(x) for x in range(1,10)]
    map = create_mask_relation_based(Path("results\\bert-base-cased\\kn"), relation_ids)
    print(map)


