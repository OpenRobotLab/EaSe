import argparse
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path

from src.dataset.datasets import ScanReferDataset, Nr3DDataset
from src.relation_encoders.base import ScanReferCategoryConcept, CategoryConcept, Near, Far
from src.relation_encoders.unary import AgainstTheWall, AtTheCorner, Tall, Low, OnTheFloor, Small, Large
from src.relation_encoders.vertical import Above, Below
from src.relation_encoders.ternary import Between as MiddleConcept
from src.relation_encoders.view import Left, Right, Front, Behind
from src.util.label import CLASS_LABELS_200
from src.util.eval_helper import get_all_categories, get_all_relations
valid_class_list = set(CLASS_LABELS_200)
valid_class_list.remove("floor")
valid_class_list.remove("wall")
valid_class_list.remove("ceiling")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

concepts_per_scene = defaultdict(dict)
concepts_classes = {
    "beside": Near,
    "near": Near,
    'next to': Near,
    "within": Near,
    "in": Near,
    "inside": Near,
    "close": Near,
    "closer": Near,
    "closest": Near,
    "far": Far,
    "farthest": Far,
    "opposite": Far,
    "furthest": Far,
    "corner": AtTheCorner,
    "against wall": AgainstTheWall,
    "above": Above,
    "on top": Above,
    'on the top': Above,
    "on": Above,
    "below": Below,
    "under": Below,
    "higher": Tall,
    "taller": Tall,
    "lower": Low,
    'beneath': Below,
    "on the floor": OnTheFloor,
    "smaller": Small,
    "shorter": Small,
    "larger": Large,
    "bigger": Large,
    "middle": MiddleConcept,
    "center": MiddleConcept,
    "between": MiddleConcept,
    'surrounded by': Near,
    'around': Near,
    "left": Left,
    "right": Right,
    "cross": Far,
    "across": Far,
    "front": Front,
    "back": Behind,
    "behind": Behind,
    "facing": Near,
    "with": Near,
    "attached": Near,
    'largest': Large,
    'highest': Tall,
    'upper': Tall,
    'longer': Large,
}

concept_num = {
    Near: 2,
    Far: 2,
    AgainstTheWall: 1,
    Above: 2,
    Below: 2,
    Tall: 1,
    Low: 1,
    Small: 1,
    Large: 1,
    MiddleConcept: 3,
    Left: 2,
    Right: 2,
    Front: 2,
    AtTheCorner: 1,
    OnTheFloor: 1,
    Behind: 2,
}

rel_num = {}
for k, v in concepts_classes.items():
    rel_num[k] = concept_num[v] - 1
    
ALL_VALID_RELATIONS = concepts_classes.keys()

def get_all_features(scan_id, concept_names: list, dataset, dataset_type, label_type):
    scan_data = dataset.scans[scan_id]
    all_concepts = {}
    if dataset_type == "scanrefer":
        category_concept = ScanReferCategoryConcept(scan_data, label_type=label_type)
    elif dataset_type == "nr3d":
        category_concept = CategoryConcept(scan_data, label_type=label_type)

    # Convert and move object locations to DEVICE only once
    object_locations = torch.from_numpy(np.vstack(scan_data["pred_locs"])).to(DEVICE)
    room_points = None  # Only load room points if needed

    for concept_name in concept_names:
        # Skip if concept_name is not valid
        if concept_name not in ALL_VALID_RELATIONS:
            # Retrieve non-relationship concepts
            all_concepts[concept_name] = category_concept.forward(concept_name)
            continue

        # Process valid relationships
        if rel_num[concept_name] == 0:
            # Special case handling for certain concepts
            if concept_name in ["corner", "against wall", "on the floor"]:
                # Load room points only once, if needed
                if room_points is None:
                    room_points = torch.from_numpy(scan_data["room_points"]).to(DEVICE)

                # Compute specific concept with room points
                all_concepts[concept_name] = (
                    concepts_classes[concept_name](object_locations, room_points)
                    .forward()
                    .float()
                )
            else:
                # Compute concept without room points, with optional convex hull (None)
                all_concepts[concept_name] = (
                    concepts_classes[concept_name](object_locations, None)
                    .forward()
                    .float()
                )
        else:
            # Compute concepts that don’t rely on additional room data
            all_concepts[concept_name] = (
                concepts_classes[concept_name](object_locations)
                .forward()
                .float()
            )

    return all_concepts
    
def compute_all_features(args):
    if args.dataset == "scanrefer":
        if args.label == "gt":
            raise NotImplementedError("Now we only support pred labels for ScanRefer.")
        dataset = ScanReferDataset()
    elif args.dataset == "nr3d":
        dataset = Nr3DDataset(label_type=args.label)
    concept_name_per_scene = defaultdict(set)
    output_path = Path(args.output)
    if args.dataset == "scanrefer":
        for i in range(len(dataset)):
            data = dataset[i]
            scan_id = data["scan_id"]
            json_obj = data["json_obj"]
            all_relations = get_all_relations(json_obj)
            all_categories = get_all_categories(json_obj)
            concept_name_per_scene[scan_id] |= set(all_relations)
            concept_name_per_scene[scan_id] |= set(all_categories)
        for scan_id, concept_names in tqdm(concept_name_per_scene.items()):
            concepts_per_scene[scan_id] = get_all_features(scan_id, concept_names, dataset, args.dataset, args.label)

        torch.save(concepts_per_scene, output_path / "scanrefer_features_per_scene.pth")
    elif args.dataset == "nr3d":
        for i, line in tqdm(enumerate(open("data/symbolic_exp/nr3d.jsonl"))):
            data = json.loads(line)
            scan_id = data["scan_id"]
            json_obj = data["json_obj"]
            all_relations = get_all_relations(json_obj)
            all_categories = get_all_categories(json_obj)
            concept_name_per_scene[scan_id] |= set(all_relations)
            concept_name_per_scene[scan_id] |= set(all_categories)

        for scan_id, concept_names in tqdm(concept_name_per_scene.items()):
            concepts_per_scene[scan_id] = get_all_features(scan_id, concept_names, dataset, args.dataset, args.label)

        torch.save(concepts_per_scene, output_path / f"nr3d_features_per_scene_{args.label}_label.pth")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["scanrefer", "nr3d"], default="scanrefer")
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--label", type=str, choices=["pred", "gt"], default="pred")
    args = args.parse_args()
    compute_all_features(args)
