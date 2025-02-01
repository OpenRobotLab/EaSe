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
from tqdm import tqdm
from PIL import Image
import os
from openai import OpenAI

from src.relation_encoders.compute_features import ALL_VALID_RELATIONS, rel_num
from src.util.eval_helper import eval_ref_one_sample, construct_bbox_corners
from src.dataset.datasets import Nr3DDataset
from src.util.vlm_utils import resize_image_to_GPT_size, encode_PIL_image_to_base64, user_prompt

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse(scan_id, json_obj, all_concepts, valid_ids):

    final_concept = torch.ones((len(valid_ids), ), device=DEVICE)
    
    appearance_concept = all_concepts[json_obj["category"]][valid_ids]
    
    if json_obj["category"] in ["corner", "middle", "room", "center"]:
        return final_concept
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
            concept = torch.ones((len(valid_ids),), device=DEVICE)
            if num == 1:
                if len(relation_item["objects"]) >= 1:
                    relation_concept = relation_concept[valid_ids, :][:, valid_ids]
                    sub_concept = parse(scan_id, relation_item["objects"][0], all_concepts, valid_ids)
                    sub_concept = relation_concept @ sub_concept
                    concept = concept * sub_concept
                else:
                    relation_concept = relation_concept[valid_ids, :][:, valid_ids]
                    obj = deepcopy(json_obj)    
                    obj["relations"] = []
                    if "color" in obj:
                        del obj["color"]
                    if "shape" in obj:
                        del obj["shape"]
                    sub_concept = parse(scan_id, obj, all_concepts, valid_ids)
                    sub_concept = relation_concept @ sub_concept
                    concept = concept * sub_concept
            elif num == 0:
                concept = relation_concept[valid_ids]
            elif num == 2:
                relation_concept = relation_concept[valid_ids, :, :][:, valid_ids, :][:, :, valid_ids]
                if len(relation_item["objects"]) == 0:
                    obj = deepcopy(json_obj)    
                    obj["relations"] = []
                    sub_concept = parse(scan_id, obj, all_concepts, valid_ids)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept, sub_concept)
                elif len(relation_item["objects"]) == 1:
                    sub_concept = parse(scan_id, relation_item["objects"][0], all_concepts, valid_ids)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept, sub_concept)
                elif len(relation_item["objects"]) == 2:
                    sub_concept_1 = parse(scan_id, relation_item["objects"][0], all_concepts, valid_ids)
                    sub_concept_2 = parse(scan_id, relation_item["objects"][1], all_concepts, valid_ids)
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
    masks_path = Path("data/nr3d_masks") / scan_id
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

def eval_nr3d(args):
    dataset = Nr3DDataset(label_type=args.label_type)
    all_features = torch.load(args.features_path)
    top_k = args.top_k
    threshold = args.threshold
    use_vlm = args.use_vlm
    scan_data = {}
    result = {
        "easy": {"correct": 0, "total": 3669},
        "hard": {"correct": 0, "total": 3816},
        "view_dependent": {"correct": 0, "total": 2478},
        "view_independent": {"correct": 0, "total": 5007},
        "overall": {"correct": 0, "total": 7485}
    }
    
    for line in tqdm(open("data/symbolic_exp/nr3d.jsonl")):
        data = json.loads(line)
        scan_id = data["scan_id"]
        json_obj = data["json_obj"]
        target_obj_id = data["target_id"]
        dataset_idx = data["dataset_idx"]
        sentence = data["sentence"]
        is_view_dependent = dataset[dataset_idx]["is_view_dependent"]   
        is_hard = dataset[dataset_idx]["is_hard"]
        
        features_this_scene = all_features[scan_id]
        scan_data = dataset.scans[scan_id]
        all_locations = scan_data["pred_locs"]
        valid_ids = torch.arange(len(all_locations)).to(DEVICE)
        try:
            final_concept = parse(scan_id, json_obj, features_this_scene, valid_ids)            
        except (NotImplementedError, ValueError, RuntimeError):
            continue
        
        max_bbox = torch.argmax(final_concept).item()
        top_k_ids = torch.topk(final_concept, min(top_k, len(final_concept))).indices
        
        gt_locs = scan_data["gt_locs"]
        pred_locs = scan_data["pred_locs"]
        obj_ids = scan_data["obj_ids"]
        assert len(final_concept) == len(obj_ids)
        target_obj_id = data["target_id"]
        obj_ids = scan_data["obj_ids"]
        gt_center, gt_size = gt_locs[target_obj_id][:3],gt_locs[target_obj_id][3:]
        answers = []
        best_iou_at_top_k = 0 
        for top_k_id in top_k_ids:
            pred = pred_locs[top_k_id]
            pred_center, pred_size = pred[:3], pred[3:]
            iou = eval_ref_one_sample(
                construct_bbox_corners(pred_center, pred_size),
                construct_bbox_corners(gt_center, gt_size)
            )
            best_iou_at_top_k = max(best_iou_at_top_k, iou)
        
        pred = all_locations[max_bbox]
        pred_center, pred_size = pred[:3], pred[3:]
        iou = best_iou_at_top_k
        top_k_candidates = [obj_ids[top_k_id] for top_k_id in top_k_ids]
        logits = final_concept[top_k_ids]
        logits /= logits.max()
        is_hard, is_view_dependent = dataset[dataset_idx]["is_hard"], dataset[dataset_idx]["is_view_dependent"]
        if use_vlm:
            answers = [target_obj_id]
            filtered_candidates = [
                candidate for candidate, logit in zip(top_k_candidates, logits) if logit > threshold
            ]
            if all([candidate in answers for candidate in filtered_candidates]):
                result["overall"]["correct"] += 1
                if is_hard:
                    result["hard"]["correct"] += 1
                else:
                    result["easy"]["correct"] += 1
                if is_view_dependent:
                    result["view_dependent"]["correct"] += 1
                else:
                    result["view_independent"]["correct"] += 1
            if all([candidate not in answers for candidate in filtered_candidates]):
                continue
            
            answer_from_vlm = query_vlm(scan_id, sentence, filtered_candidates)
            if answer_from_vlm in answers:
                result["overall"]["correct"] += 1
                if is_hard:
                    result["hard"]["correct"] += 1
                else:
                    result["easy"]["correct"] += 1
                if is_view_dependent:
                    result["view_dependent"]["correct"] += 1
                else:
                    result["view_independent"]["correct"] += 1
        else:
            if top_k_candidates[0] == target_obj_id:
                if is_hard:
                    result["hard"]["correct"] += 1
                else:
                    result["easy"]["correct"] += 1
                if is_view_dependent:
                    result["view_dependent"]["correct"] += 1
                else:
                    result["view_independent"]["correct"] += 1
                result["overall"]["correct"] += 1
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
    parser.add_argument("--label_type", choices=["pred", "gt"], type=str)
    args = parser.parse_args()
    eval_nr3d(args)
