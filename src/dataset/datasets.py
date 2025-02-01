from copy import deepcopy
from collections import Counter
import json
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import multiprocessing as mp
from src.util.eval_helper import convert_pc_to_box, is_explicitly_view_dependent, load_pc, \
SCANNET_CLASS_LABELS_200, scan_id_to_idx, eval_ref_one_sample, construct_bbox_corners

SCAN_FAMILY_BASE = Path("data/referit3d")
pc_path = SCAN_FAMILY_BASE / "scan_data" / "pcd_with_global_alignment"
Path("data/referit3d/scan_data/pcd_with_global_alignment")

int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
cat2int = {w: i for i, w in enumerate(int2cat)}

with open('data/referit3d/feats_3d.pkl', 'rb') as f:
    feats = pickle.load(f)
    
def load_pc_(scan_id):
    obj_ids = feats[scan_id]['obj_ids']
    inst_locs = feats[scan_id]['inst_locs']
    center = feats[scan_id]['center']
    obj_embeds = feats[scan_id]['obj_embeds']

    return obj_ids, inst_locs, center, obj_embeds

from copy import deepcopy
def load_pred_ins(scan_id, normalize=True, use_scannet200=False):
    idx = scan_id_to_idx[scan_id]
    odin_output = torch.load(f"data/seg/outputs_{idx}.pt")[0]['instances_3d']
    pred_classes = odin_output['pred_classes'].cpu()
    pred_scores = odin_output['pred_scores'].cpu() # (100, )
    pred_masks = odin_output['pred_masks'].cpu()
    # get the order of pred_scores 
    pred_scores_cp = deepcopy(pred_scores)
    _, order = pred_scores_cp.sort(descending=True)
    order = order.tolist()
    assert set(order) == set(range(100))
    assert pred_scores[order[0]] >= pred_scores[order[1]]
    pcds, colors, _, instance_labels = torch.load(SCAN_FAMILY_BASE / "scan_data"/ "pcd_with_global_alignment" / f"{scan_id}.pth")
    obj_ids = set()
    batch_pcds = []
    inst_locs = []
    scene_pc = []
    batch_labels = []
    selected_idx = []
    for i in range(100):
        scores = pred_scores[i]
        if scores < 0.45:
            continue
        label = SCANNET_CLASS_LABELS_200[pred_classes[i] - 1]
        if label in ['wall', 'floor', 'ceiling']:
            continue
        mask = pred_masks[:, i]
        obj_pcd = pcds[mask]
        
        scene_pc.append(obj_pcd)
        box = convert_pc_to_box(obj_pcd)
        
        if inst_locs and max([
            eval_ref_one_sample(construct_bbox_corners(*box), construct_bbox_corners(bbox[:3], bbox[3:]))
            for bbox in inst_locs
        ]) > 0.9:
            continue
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))
        obj_ids.add(pred_classes[i].item())
        obj_pcd = obj_pcd - obj_pcd.mean(0)
        max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
        if max_dist < 1e-6:
            max_dist = 1
        obj_pcd = obj_pcd / max_dist
        obj_color = colors[mask] / 127.5 - 1
        obj_height = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()
        obj_pcd = obj_pcd[:, :3]
        pcd_idxs = np.random.choice(len(obj_pcd), size=2048, replace=(len(obj_pcd) < 2048))
        pcd_idxs = np.arange(len(obj_pcd))
        obj_pcd = obj_pcd[pcd_idxs]
        obj_color = obj_color[pcd_idxs]
        obj_height = obj_height[pcd_idxs]
        batch_pcds.append(np.concatenate([
            obj_pcd,
            obj_height,
            obj_color,
        ], 1))
        batch_labels.append(label)
        selected_idx.append(i)
    
    scene_pc = np.concatenate(scene_pc, 0)

    return batch_labels, inst_locs, selected_idx, batch_pcds

# def load_one_scene(scan_id, label_type):
#     inst_labels = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_name', '%s.json'%scan_id)))
#     inst_locs = np.load(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_loc', '%s.npy'%scan_id))
#     inst_colors = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_gmm_color', '%s.json'%scan_id)))
#     result = {
#         'gt_locs': inst_locs,     
#         'gt_colors': inst_colors, 
#         "scan_id": scan_id
#     }
#     pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", '%s.pth'% scan_id))
    
#     points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    
#     result["room_points"] = points
#     center, size = convert_pc_to_box(points)
#     corners = np.array([
#         [center[0] + size[0] / 2, center[1] + size[1] / 2],
#         [center[0] + size[0] / 2, center[1] - size[1] / 2],
#         [center[0] - size[0] / 2, center[1] - size[1] / 2],
#         [center[0] - size[0] / 2, center[1] + size[1] / 2],
#     ])
#     result["room_corners"] = corners
#     info = load_pc_(scan_id)
#     obj_ids, pred_inst_locs, center, obj_embeds = info
#     result["obj_ids"] = obj_ids
#     if label_type == "gt":
#         result["pred_locs"] = [inst_locs[i] for i in obj_ids]
#         result["inst_labels"] = [inst_labels[i] for i in obj_ids]
#     else:
#         result["pred_locs"] = pred_inst_locs
#         result["obj_embeds"] = obj_embeds

#     pcds = points
#     obj_pcds = []
#     all_colors = []
#     for i in range(instance_labels.max() + 1):
#         mask = instance_labels == i     
#         obj_pcds.append(torch.from_numpy(pcds[mask]).cuda())
#         all_colors.append(torch.from_numpy(colors[mask]).cuda())
#     result['pcds'] = obj_pcds                     
#     all_colors = [color.mean(dim=0) for color in all_colors]
#     result['colors'] = all_colors
#     return scan_id, result

def load_one_scene(scan_id, label_type):
    inst_labels = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_name', '%s.json'%scan_id)))
    inst_locs = np.load(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_loc', '%s.npy'%scan_id))
    inst_colors = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_gmm_color', '%s.json'%scan_id)))
    result = {
        'gt_locs': inst_locs,     
        'gt_colors': inst_colors, 
        "scan_id": scan_id
    }
    pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", '%s.pth'% scan_id))
    
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    
    result["room_points"] = points
    center, size = convert_pc_to_box(points)
    corners = np.array([
        [center[0] + size[0] / 2, center[1] + size[1] / 2],
        [center[0] + size[0] / 2, center[1] - size[1] / 2],
        [center[0] - size[0] / 2, center[1] - size[1] / 2],
        [center[0] - size[0] / 2, center[1] + size[1] / 2],
    ])
    result["room_corners"] = corners
    info = load_pc_(scan_id)
    obj_ids, pred_inst_locs, center, obj_embeds = info
    result["obj_ids"] = obj_ids
    if label_type == "gt":
        result["pred_locs"] = pred_inst_locs # USING GT LABEL
        result["inst_labels"] = [inst_labels[i] for i in obj_ids]
    else:
        result["pred_locs"] = pred_inst_locs
        result["inst_labels"] = [inst_labels[i] for i in obj_ids]
    result["obj_embeds"] = obj_embeds
    # walls = [
    #     inst_locs[i] for i in range(len(inst_labels)) if inst_labels[i] == "wall"
    # ]
    # result["walls"] = walls
    pcds = points
    obj_pcds = []
    all_colors = []
    for i in range(instance_labels.max() + 1):
        mask = instance_labels == i     
        obj_pcds.append(torch.from_numpy(pcds[mask]).cuda())
        all_colors.append(torch.from_numpy(colors[mask]).cuda())
    result['pcds'] = obj_pcds                     
    all_colors = [color.mean(dim=0) for color in all_colors]
    result['colors'] = all_colors
    return scan_id, result

def load_scannet(scan_ids, label_type):
    scans = {}
    for scan_id in tqdm(scan_ids):
        scan_id, result = load_one_scene(scan_id, label_type)
        scans[scan_id] = result
    return scans

class Nr3DDataset:
    def __init__(self, 
            anno_type='nr3d', 
            split="test",
            label_type="pred"
        ):
        assert anno_type in ['nr3d', 'sr3d']
        assert label_type in ['gt', 'pred']
        if split == "test":
            anno_file = SCAN_FAMILY_BASE / f'annotations/refer/{anno_type}_test.csv'
        elif split == "train":
            anno_file = SCAN_FAMILY_BASE / f'annotations/refer/{anno_type}.csv'
        self.scan_ids = set() # scan ids in data
        self.data = [] 
        training_ids = open(SCAN_FAMILY_BASE / "annotations/splits/scannetv2_train.txt").read().splitlines()
        if split == "train":
            df = pd.read_csv(anno_file)
            for i in range(len(df)):
                mention = df.iloc[i]['mentions_target_class']
                use_color = df.iloc[i]['uses_color_lang']
                use_shape = df.iloc[i]['uses_shape_lang']
                if not mention:
                    continue
                if split == "train" and (use_color or use_shape):
                    continue
                if split == "train" and df.iloc[i]['scan_id'] not in training_ids:
                    continue
                self.scan_ids.add(df.iloc[i]['scan_id'])
                item = df.iloc[i]
                self.data.append(item)
        else:
            anno_file = SCAN_FAMILY_BASE / f'annotations/refer/{anno_type}_test.csv'
            df = pd.read_csv(anno_file)
            for i in range(len(df)):
                self.scan_ids.add(df.iloc[i]['scan_id'])
                item = df.iloc[i]
                self.data.append(item)

        # load category file
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.scans = load_scannet(self.scan_ids, label_type)
        
        # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scans[scan_id]['inst_labels']
            self.scans[scan_id]['label_count'] = Counter([l for l in inst_labels])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['stimulus_id']
        distractor_ids = list(map(int, item_id.split('-')[4:]))
        scan_id = item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        tgt_object_label = obj_labels[self.scans[scan_id]['obj_ids'].index(tgt_object_id)]
        sentence = item['utterance']
        if type(item["tokens"]) == str:
            item["tokens"] = eval(item["tokens"])        
        is_view_dependent = is_explicitly_view_dependent(item['tokens'])
            
        
        is_multiple = self.scans[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scans[scan_id]['label_count'][tgt_object_label] > 2
        
        
        data_dict = {
            "scan_id": scan_id,
            "sentence": sentence,
            "tgt_object_id": int(tgt_object_id), # 1
            "tgt_object_label": tgt_object_label,
            "data_idx": item_id,
            "distractors": distractor_ids,
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard,
        }
    
        return data_dict


label_map_file = 'data/referit3d/annotations/meta_data/scannetv2-labels.combined.tsv'
labels_pd = pd.read_csv(label_map_file, sep='\t', header=0)
class ScanReferDataset:
    def __init__(self):
        self.scan_ids = set() # scan ids in data
        self.data = [] 
        eval_data = json.load(open("data/symbolic_exp/scanrefer.json", "r"))
        for d in eval_data:
            self.scan_ids.add(d["scan_id"])
            self.data.append(d)
        # load category file
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.scans = {}
        self.scans = torch.load(SCAN_FAMILY_BASE / "scan_data" / "scan_data.pth")
        for scan_id in self.scan_ids:
            inst_labels = self.scans[scan_id]['inst_labels']
            self.scans[scan_id]['label_count'] = Counter([l for l in inst_labels])
        
    def _load_one_scene(self, scan_id):
        inst_labels = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_name', '%s.json'%scan_id)))
        inst_locs = np.load(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_loc', '%s.npy'%scan_id))
        inst_colors = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_gmm_color', '%s.json'%scan_id)))
        result = {
            'inst_labels': inst_labels,
            'gt_locs': inst_locs,     
            'gt_colors': inst_colors, 
            "scan_id": scan_id
        }
        pred_labels, pred_locs, selected_idx, pred_pcds = load_pred_ins(scan_id, use_scannet200=True)
        result["pred_labels"] = pred_labels
        result["pred_locs"] = pred_locs
        result["pcds"] = pred_pcds
        
        pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", '%s.pth'% scan_id))
        batch_labels, obj_ids, inst_locs, center, batch_pcds = load_pc(scan_id)
        batch_class_ids = []

        for obj_label in batch_labels:
            label_ids = labels_pd[labels_pd['raw_category'] == obj_label]['nyu40id']
            label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
            batch_class_ids.append(label_id)
        result['obj_ids'] = obj_ids
        result['inst_locs'] = inst_locs
        result['batch_class_ids'] = batch_class_ids
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        
        result["room_points"] = points
        result['selected_idx'] = selected_idx
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # load scanrefer
        item = self.data[idx]
        json_obj = item["json_obj"]
        scan_id = item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_bbox = self.scans[scan_id]['gt_locs'][tgt_object_id]
        obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        tgt_object_label = obj_labels[tgt_object_id]
        
        is_multiple = self.scans[scan_id]['label_count'][tgt_object_label] > 1
        
        data_dict = {
            "scan_id": scan_id,
            "tgt_object_id": int(tgt_object_id), # 1
            "tgt_bbox": tgt_bbox,
            "tgt_object_label": tgt_object_label,
            'is_multiple': is_multiple,
            "json_obj": json_obj,
            "caption": item["caption"]
        }
    
        return data_dict

if __name__ == "__main__":
    dataset = Nr3DDataset()
    dataset = ScanReferDataset()