from copy import deepcopy
import cv2
import json
from json import JSONDecodeError
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from skimage import img_as_ubyte
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
from openai import OpenAI


from src.relation_encoders.compute_features import ALL_VALID_RELATIONS, rel_num
from src.util.eval_helper import eval_ref_one_sample, construct_bbox_corners, load_pc
from src.dataset.datasets import ScanReferDataset
from src.util.vlm_utils import resize_image_to_GPT_size, encode_PIL_image_to_base64, user_prompt

dataset = ScanReferDataset()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse(scan_id, json_obj, all_concepts):
    appearance_concept = all_concepts[json_obj["category"]]
    final_concept = torch.ones(appearance_concept.shape[0], ).to(DEVICE)
    if json_obj["category"] in ["corner", "middle", "room", "center"]:
        return torch.ones(appearance_concept.shape[0], ).to(DEVICE)
    if json_obj["category"] in ALL_VALID_RELATIONS:
        raise NotImplementedError
    
    final_concept = torch.minimum(final_concept, appearance_concept)
    concept = None

    if "relations" in json_obj:
        for relation_item in json_obj["relations"]:
            if "anchors" in relation_item.keys():
                relation_item["objects"] = relation_item["anchors"] 
            relation_name = relation_item["relation_name"]
            if relation_name not in ALL_VALID_RELATIONS:
                continue
            relation_concept = all_concepts[relation_name]
            num = rel_num[relation_name]
            
            concept = torch.ones((len(relation_concept),), device=DEVICE)
            if num == 1:
                if len(relation_item["objects"]) >= 1:
                    sub_concept = parse(scan_id, relation_item["objects"][0], all_concepts)
                    sub_concept = relation_concept @ sub_concept
                    concept = concept * sub_concept
                else:
                    obj = deepcopy(json_obj)    
                    obj["relations"] = []
                    if "color" in obj:
                        del obj["color"]
                    if "shape" in obj:
                        del obj["shape"]
                    sub_concept = parse(scan_id, obj, all_concepts)
                    sub_concept = relation_concept @ sub_concept
                    concept = concept * sub_concept
            elif num == 0:
                concept = relation_concept    
            elif num == 2:
                if len(relation_item["objects"]) == 0:
                    obj = deepcopy(json_obj)    
                    obj["relations"] = []
                    sub_concept = parse(scan_id, obj, all_concepts)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept, sub_concept)
                elif len(relation_item["objects"]) == 1:
                    sub_concept = parse(scan_id, relation_item["objects"][0], all_concepts)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept, sub_concept)
                elif len(relation_item["objects"]) == 2:
                    sub_concept_1 = parse(scan_id, relation_item["objects"][0], all_concepts)
                    sub_concept_2 = parse(scan_id, relation_item["objects"][1], all_concepts)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept_1, sub_concept_2)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
            concept = F.softmax(concept, dim=0)
            if relation_item.get("negative") == True:
                concept = concept.max() - concept
            final_concept = final_concept * concept

    return final_concept

def query_vlm(scan_id, caption, filtered_candidates):
    image_root = Path('data/frames')
    masks_path = Path("data/scanreder_masks") / scan_id
    image_dirs = os.listdir(masks_path)
    base64Frames = []
    merged_indices = {}
    for obj_name in image_dirs:
        obj_id = int(obj_name.split("_")[0])
        if obj_id not in filtered_candidates:
            continue
        indices = np.load(masks_path / obj_name / "indices.npz")
        for k in indices.keys():
            img_name = k
            if img_name not in merged_indices:
                merged_indices[img_name] = {}
            merged_indices[img_name][obj_id] = indices[k]
    
    merged_areas = {}
    for img_name in merged_indices.keys():
        img = cv2.imread(str(image_root / scan_id / "color" / img_name))
        area = 0
        for obj_id in merged_indices[img_name].keys():
            indices = merged_indices[img_name][obj_id]
            area += (indices[1].max() - indices[1].min()) * (indices[0].max() - indices[0].min())
        merged_areas[img_name] = area
    
    # top 8 images on area
    sorted_images = sorted(merged_areas.items(), key=lambda x: x[1], reverse=True)
    selected_candidates = []

    top8_images = sorted_images[:8]
    for img_name, area in top8_images:
        for obj_id in merged_indices[img_name].keys():
            selected_candidates.append(obj_id)
    # if not all objects is selected, add to the top 8 images
    length = 8
    for obj_id in filtered_candidates:
        if obj_id not in selected_candidates:
            for sorted_img_name, _ in sorted_images:
                if obj_id in merged_indices[sorted_img_name]:
                    top8_images.append((sorted_img_name, merged_areas[sorted_img_name]))
                    length -= 1
                    break
    if length != 8:
        top8_images = top8_images[:length] + top8_images[-(8 - length):]
    assert len(top8_images) <= 8
    single_img_size = img.shape[:2]
    stitched_img = np.ones((single_img_size[0] * 2, single_img_size[1] * 4, 3), dtype=np.uint8) * 255
    for i in range(2):
        for j in range(4):
            if i * 4 + j >= len(top8_images):
                continue
            img_name, area = top8_images[i * 4 + j]
            img = cv2.imread(str(image_root / scan_id / "color" / img_name))
            for obj_id, area in merged_indices[img_name].items():
                indices = merged_indices[img_name][obj_id]
                cv2.putText(img, f"obj_{obj_id}", (int((indices[1].max() + indices[1].min()) / 2) , int((indices[0].max() + indices[0].min()) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stitched_img[i * single_img_size[0]: (i + 1) * single_img_size[0], j * single_img_size[1]: (j + 1) * single_img_size[1]] = img
    stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_as_ubyte(stitched_img))
    image = resize_image_to_GPT_size(image)

    ecd = encode_PIL_image_to_base64(image)
    base64Frames.append(ecd)
    assert len(base64Frames) == 1
    
    messages = [
        {
            "role": "system",
            "content": "You are good at finding objects specified by a description in indoor rooms by watching the videos scanning the rooms."
        },
        {
            "role": "user",
            "content": None,
        }
    ]  
    messages[1]["content"] = [
        {
            "type": "text",
            "text": user_prompt.format(utterance=caption, candidates=str(filtered_candidates))
        },
        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "high"}}, base64Frames),
    ]
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": messages,
        "top_p": 0.3,
        "temperature": 0.1,
    }
    cv2.imwrite("tmp.jpg", stitched_img)
    exit()
    openai_response = client.chat.completions.create(
        **payload
    ).choices[0].message.content
    
    try:
        if "```json" in openai_response:
            openai_response = openai_response.split("```json")[1].split("```")[0].strip()
        json_obj = json.loads(openai_response)
        return int(json_obj['object id'])
    except (JSONDecodeError, KeyError):
        return -1

def eval_scanrefer(args):
    all_concepts = torch.load(args.features_path)
    skip_undet_examples = False # if True, skip examples where the target object is not detected
    top_k = 1
    scan_data = {}
    for scan_id in tqdm(dataset.scan_ids):
        gt_labels, obj_ids, gt_locs, _, _ = load_pc(scan_id)
        scan_data[scan_id] = {
            "gt_labels": gt_labels,
            "obj_ids": obj_ids,
            "gt_locs": gt_locs,
        }
    label_map_file = 'data/referit3d/annotations/meta_data/scannetv2-labels.combined.tsv'
    labels_pd = pd.read_csv(label_map_file, sep='\t', header=0)
    skip = 0
    result = {
        "single_25": {"correct": 0, "total": 0},
        "multiple_25": {"correct": 0, "total": 0},
        "total_25": {"correct": 0, "total": 0},
    }

    batch_class_ids_per_scan = {}
    for scan_id in tqdm(dataset.scan_ids):
        gt_labels = scan_data[scan_id]['gt_labels']
        batch_class_ids = []
        for obj_label in gt_labels:
            label_ids = labels_pd[labels_pd['raw_category'] == obj_label]['nyu40id']
            label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
            batch_class_ids.append(label_id)
        batch_class_ids_per_scan[scan_id] = batch_class_ids

    for i in tqdm(range(len(dataset))):
        scan_id = dataset[i]["scan_id"]
        json_obj = dataset[i]["json_obj"]
        concept_this_scene = all_concepts[scan_id]
        try:
            final_concept = parse(scan_id, json_obj, concept_this_scene)
            assert len(final_concept.shape) == 1
            top_k_ids = torch.topk(final_concept, top_k).indices
        except (NotImplementedError, ValueError, RuntimeError):
            continue
        target_obj_id = dataset[i]["tgt_object_id"]
        pred_locs = dataset.scans[scan_id]["pred_locs"]
        inst_locs = dataset.scans[scan_id]['gt_locs']
        obj_ids = scan_data[scan_id]['obj_ids']
        selected_idx = dataset.scans[scan_id]['obj_ids']
        target_box = scan_data[scan_id]['gt_locs'][scan_data[scan_id]['obj_ids'].index(target_obj_id)]
        gt_center = target_box[:3]
        gt_size = target_box[3:]
        index = obj_ids.index(target_obj_id)
        batch_class_ids = batch_class_ids_per_scan[scan_id]
        target_class_id = batch_class_ids[index]
        unique = (np.array(batch_class_ids) == target_class_id).sum() == 1
        if skip_undet_examples:
            best_iou = max([
                eval_ref_one_sample(
                    construct_bbox_corners(pred_locs[i][:3], pred_locs[i][3:]),
                    construct_bbox_corners(gt_center, gt_size)
                )
                for i in range(len(pred_locs))
            ])
            if best_iou < 0.25:
                skip += 1
                continue
        best_iou_at_top_k = 0
        if not args.use_vlm:
            pred = pred_locs[top_k_ids[0]]
            pred_center, pred_size = pred[:3], pred[3:]
            iou = eval_ref_one_sample(
                construct_bbox_corners(pred_center, pred_size),
                construct_bbox_corners(gt_center, gt_size)
            )
            if iou >= 0.25:
                if unique:
                    result["single_25"]["correct"] += 1
                else:
                    result["multiple_25"]["correct"] += 1
                result["total_25"]["correct"] += 1
            if unique:
                result["single_25"]["total"] += 1
            else:
                result["multiple_25"]["total"] += 1
            continue
        else:
            answers = []
            for top_k_id in top_k_ids:
                pred = pred_locs[top_k_id]
                pred_center, pred_size = pred[:3], pred[3:]
                iou = eval_ref_one_sample(
                    construct_bbox_corners(pred_center, pred_size),
                    construct_bbox_corners(gt_center, gt_size)
                )
                best_iou_at_top_k = max(best_iou_at_top_k, iou)
                if iou >= 0.25:
                    answers.append(selected_idx[top_k_id])
            top_k_candidates = [selected_idx[top_k_id] for top_k_id in top_k_ids]
            logits = final_concept[top_k_ids]
            assert len(top_k_candidates) == len(logits)
            if len(answers) == 0:
                continue
            for answer in answers:
                assert answer in top_k_candidates
            logits /= logits.max()
        
            pred_obj_id = query_vlm(scan_id, dataset[i]["caption"], top_k_candidates)
            if pred_obj_id in answers:
                if unique:
                    result["single_25"]["correct"] += 1
                else:
                    result["multiple_25"]["correct"] += 1
                result["total_25"]["correct"] += 1
            if unique:
                result["single_25"]["total"] += 1
            else:
                result["multiple_25"]["total"] += 1

    result["total_25"]["total"] = 9507
    
    for k, v in result.items():
        if v["total"]:
            print(k, v["correct"], v["total"], v["correct"] / v["total"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--use_vlm", action="store_true")
    args = parser.parse_args()
    eval_scanrefer(args)