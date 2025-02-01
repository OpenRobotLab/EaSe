import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json

nr3d = pd.read_csv("data/referit3d/annotations/refer/nr3d.csv")
import pickle
lookuptable = defaultdict(dict)
instance_loc_table = {}
instance_name_table = {}
room_point_table = {}
scan_ids = set()


with open("data/tables", "rb") as f:
    lookuptable, instance_loc_table, instance_name_table, room_point_table = pickle.load(f)

print("Data loaded")
import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function

class_name = None

def extract_class_name(file_path):
    with open(file_path) as f:
        for line in f:
            if "class" in line:
                return line.split("class")[1].split(":")[0].strip()



correct = 0

def test_code(relation_name, file_path):
    training_cases = json.load(open(f"data/test_data/{relation_name}/cases.json"))
    if "mps.matmul" in open(file_path).read():
        return 0, 1, []
    fail_cases = []
    class_name = extract_class_name(file_path)
    Concept = import_class_from_file(file_path, class_name)
    correct, total = 0, 0
    for training_data in training_cases[:100]:
        pred_locs = instance_loc_table[training_data["scan_id"]]
        instance_names = json.load(open(f"data/referit3d/scan_data/instance_id_to_name/{training_data['scan_id']}.json"))
        dataset_idx = lookuptable[training_data["scan_id"]][training_data["sentence"]]
        stimulus_id = nr3d.iloc[dataset_idx]["stimulus_id"].split("-")
        if not (training_data['scan_id'] == stimulus_id[0] and training_data["target_category"] == stimulus_id[1]):
            continue
        candidates = [int(i) for i in stimulus_id[3:]]
        json_obj = training_data["json_obj"]
        if nr3d.iloc[dataset_idx]["correct_guess"] != True:
            continue    
        target_id = int(nr3d.iloc[dataset_idx]["target_id"])
        target_label = training_data["target_category"]
        if target_label not in instance_names:
            continue
        if len(json_obj["relations"][0]["objects"]) > 0:
            anchor_label = json_obj["relations"][0]["objects"][0]["category"]
        else:
            anchor_label = target_label
        if anchor_label not in instance_names:
            continue
        anchor_ids = [i for i, name in enumerate(instance_names) if anchor_label == name]
        
        if relation_name in ['left', 'right']:
            assert sorted(candidates) == anchor_ids
            assert len(candidates) == len(anchor_ids) == 2
            concept = Concept(pred_locs).forward()
            anchor_ids.remove(target_id)
            anchor_id = anchor_ids[0]
            scan_id = training_data["scan_id"]
            if concept[target_id, anchor_id] > 0 and concept[anchor_id, target_id] <= 0:
                correct += 1
            else:
                fail_cases.append(
                    {   
                        "scan_id": training_data["scan_id"],
                        "sentence": training_data["sentence"],
                        "target": target_id,
                        "target_anchor": anchor_id,
                        "pred": anchor_id,
                        "pred_anchor": target_id,
                    }
                )
        
        else:
            above_concept = Concept(pred_locs).forward()
            if len(anchor_ids) != 1:
                continue
            anchor = anchor_ids[0]
            fail = False
            assert target_id not in candidates[1:]
            for candidate in candidates[1:]:
                if above_concept[target_id, anchor] <= 0 or above_concept[candidate, anchor] > above_concept[target_id, anchor]:
                    fail = True
                    fail_cases.append(
                        {   
                            "scan_id": training_data["scan_id"],
                            "sentence": training_data["sentence"],
                            "target": target_id,
                            "target_anchor": anchor,
                            "pred": candidate,
                            "pred_anchor": anchor,
                        }
                    )
                    break
            if not fail:
                correct += 1
            
        total += 1

    
    assert len(fail_cases) == total - correct
    return correct, total, fail_cases 

def test_code_for_front(relation_name, file_path):
    training_cases = json.load(open(f"data/test_data/{relation_name}/cases.json"))
    fail_cases = []
    class_name = extract_class_name(file_path)
    if "mps.matmul" in open(file_path).read():
        return 0, 1, []
    try:
        Concept = import_class_from_file(file_path, class_name)
    except AttributeError:
        Concept = import_class_from_file(file_path, 'Front')

    front_concepts = {}
    correct, total = 0, 0
    for training_data in tqdm(training_cases):
        pred_locs = instance_loc_table[training_data["scan_id"]]
        front_concept = Concept(pred_locs).forward()
        front_concepts[training_data['scan_id']] = front_concept
        setence = training_data["sentence"]
        dataset_idx = lookuptable[training_data["scan_id"]][training_data["sentence"]]
        stimulus_id = nr3d.iloc[dataset_idx]["stimulus_id"].split("-")
        assert training_data['scan_id'] == stimulus_id[0]
        candidates = [int(i) for i in stimulus_id[4:]]
        
        if nr3d.iloc[dataset_idx]["correct_guess"] != True:
            continue    
        target_id = int(nr3d.iloc[dataset_idx]["target_id"])
        assert target_id == training_data["target"]
        target = int(training_data["target"])
        assert target not in candidates
        anchor = training_data["anchor"]
        instance_names = instance_name_table[training_data["scan_id"]]
        assert sum([anchor == category for category in instance_names]) == 1
        anchor = instance_names.index(anchor)
        fail = False

        for distractor in candidates:
            distractor = int(distractor)
            if front_concept[target, anchor] <= 0 or front_concept[distractor, anchor] > front_concept[target, anchor]:
                fail = True
                fail_cases.append(
                    {   
                        "scan_id": training_data["scan_id"],
                        "sentence": " ",
                        "target": target,
                        "target_anchor": anchor,
                        "pred": distractor,
                        "pred_anchor": anchor,
                    }
                )
        if not fail:
            correct += 1            
        total += 1


    return correct, total, fail_cases

def test_code_for_behind(relation_name, file_path):
    training_cases = json.load(open(f"data/test_data/{relation_name}/cases.json"))
    fail_cases = []
    class_name = extract_class_name(file_path)
    if "matmul" in open(file_path).read():
        return 0, 1, []
    try:
        Concept = import_class_from_file(file_path, class_name)
    except AttributeError:
        Concept = import_class_from_file(file_path, 'Behind')
    Concept = import_class_from_file(file_path, 'Behind')
    front_concepts = {}
    correct, total = 0, 0
    for training_data in tqdm(training_cases[:50]):
        pred_locs = instance_loc_table[training_data["scan_id"]]
        front_concept = Concept(pred_locs).forward()
        front_concepts[training_data['scan_id']] = front_concept
        setence = training_data["utterance"]
        distractors = training_data["distactors"]
        target = training_data["target"]
        anchor = training_data["anchor"]
        fail = False
        for distractor in distractors:
            distractor = int(distractor)
            if front_concept[target, anchor] <= 0 or front_concept[distractor, anchor] > front_concept[target, anchor]:
                fail = True
                fail_cases.append(
                    {   
                        "scan_id": training_data["scan_id"],
                        "sentence": setence,
                        "target": target,
                        "target_anchor": anchor,
                        "pred": distractor,
                        "pred_anchor": anchor,
                    }
                )
        if not fail:
            correct += 1            
        total += 1

    return correct, total, fail_cases

def test_code_for_between(relation_name, file_path):
    training_cases = json.load(open(f"data/test_data/{relation_name}/cases.json"))
    if "matmul" in open(file_path).read():
        return 0, 1, []
    fail_cases = []
    correct, total = 0, 0
    class_name = extract_class_name(file_path)
    Concept = import_class_from_file(file_path, class_name)
    for case in tqdm(training_cases):
        scan_id = case["scan_id"]
        pred_locs = instance_loc_table[case["scan_id"]]
        dataset_idx = lookuptable[case["scan_id"]][case["sentence"]]
        stimulus_id = nr3d.iloc[dataset_idx]["stimulus_id"].split("-")
        candidates = [int(i) for i in stimulus_id[4:]]
        
        target_type = nr3d.iloc[dataset_idx]["instance_type"]
        json_obj = case["json_obj"]
        if json_obj["category"] != target_type:
            continue
        instance_names = instance_name_table[scan_id]
        if sum([category == target_type for category in instance_names]) != 3:
            continue
        assert len(candidates) == 2
        target_id = int(nr3d.iloc[dataset_idx]["target_id"])
        concept = Concept(pred_locs).forward()
        anchor_1, anchor_2 = candidates
        if concept[target_id, anchor_1, anchor_2] < concept[anchor_1, target_id, anchor_2] or \
            concept[target_id, anchor_1, anchor_2] < concept[anchor_2, target_id, anchor_1] or \
                concept[target_id, anchor_1, anchor_2] <= 0:
                fail_cases.append(
                    {   
                        "scan_id": scan_id,
                        "sentence": case["sentence"],
                        "target": target_id,
                        "target_anchor": anchor_1,
                        "pred": anchor_2,
                        "pred_anchor": -1
                    }
                )
        else:
            correct += 1
        total += 1

    return correct, total, fail_cases

def test_code_for_unary(relation_name, file_path):
    training_cases = json.load(open(f"data/test_data/{relation_name}/cases.json"))
    correct, total = 0, 0
    fail_cases = []
    class_name = extract_class_name(file_path)
    
    Concept = import_class_from_file(file_path, class_name)
    for training_data in tqdm(training_cases):
        pred_locs = instance_loc_table[training_data["scan_id"]]
        room_point = room_point_table[training_data["scan_id"]]
        total += 1
        unary_concept = Concept(pred_locs, room_point).forward()
        assert unary_concept.shape[0] == pred_locs.shape[0]
        
        setence = training_data["sentence"]
        dataset_idx = lookuptable[training_data["scan_id"]][training_data["sentence"]]
        stimulus_id = nr3d.iloc[dataset_idx]["stimulus_id"].split("-")
        assert training_data['scan_id'] == stimulus_id[0]
        candidates = [int(i) for i in stimulus_id[4:]]
        target_id = int(nr3d.iloc[dataset_idx]["target_id"])
        assert target_id not in candidates
        fail = False
        for candidate in candidates:
            candidate = int(candidate)
            if unary_concept[candidate] > unary_concept[target_id] or unary_concept[target_id] <= 0:
                fail = True
                fail_cases.append(
                    {   
                        "scan_id": training_data["scan_id"],
                        "sentence": setence,
                        "target": target_id,
                        "pred": candidate,
                        "target_anchor": 0,
                        "pred_anchor": 0,
                    }
                )
        if not fail:
            correct += 1
    return correct, total, fail_cases
