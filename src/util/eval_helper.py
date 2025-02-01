# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import json

def get_all_keys(d):
    all_keys = set()
    for k, v in d.items():
        all_keys.add(k)
        if isinstance(v, dict):
            all_keys |= get_all_keys(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    all_keys |= get_all_keys(item)
    return all_keys

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]
    
    return x_min, x_max, y_min, y_max, z_min, z_max

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou

def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou

def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size

def is_explicitly_view_dependent(tokens):
    """
    :return: a boolean mask
    """
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    for token in tokens:
        if token in target_words:
            return True
    return False

color_to_rgb = {
  "beige": [245, 245, 220],
  "black": [0, 0, 0],
  "blue": [0, 0, 255],
  "brown": [165, 42, 42],
  "golden": [255, 215, 0],
  "green": [0, 255, 0],
  "red": [255, 0, 0],
  "white": [255, 255, 255],
  "yellow": [255, 255, 0]
}

def find_color(rgb: np.array):
    
    rgb *= 255.0
    
    rgb = rgb.mean(axis=0).reshape(3)
    print(rgb)
    assert rgb.shape == (3,)
    assert rgb.max() <= 255.0 and rgb.min() >= 0.0
    min_dist = 1e9
    color = None
    for k, v in color_to_rgb.items():
        dist = np.linalg.norm(np.array(v) - rgb)
        if dist < min_dist:
            min_dist = dist
            color = k
    return color

def nearest_neighbor(x, y):
    """
    :param x: a tensor of shape (N, 3)
    :param y: a tensor of shape (M, 3)
    :return: Euclidean distance between the nearest neighbors of x and y
    """
    x = x[:, :3]
    y = y[:, :3]
    distances = torch.cdist(x, y, p=2)
    
    # Find the minimum distance
    min_distance, _ = torch.min(distances, dim=1)
    
    # Return the minimum of these minimum distances
    return torch.min(min_distance)


def get_all_appearance(json_obj):
    appearances = set()
    appearances.add(json_obj["appearance"])
    if "relations" in json_obj:
        for relation in json_obj["relations"]:
            if "anchors" in relation:
                relation["objects"] = relation["anchors"]
            for obj in relation["objects"]:
                appearances |= get_all_appearance(obj)
    return appearances

def get_all_categories(json_obj):
    appearances = set()
    appearances.add(json_obj["category"])
    if "relations" in json_obj:
        for relation in json_obj["relations"]:
            if "anchors" in relation:
                relation["objects"] = relation["anchors"]
            for obj in relation["objects"]:
                appearances |= get_all_categories(obj)
    return appearances

def get_all_relations(json_obj):
    relations = set()
    if "relations" in json_obj:
        for relation in json_obj["relations"]:
            relations.add(relation["relation_name"])
            if "anchors" in relation:
                relation["objects"] = relation["anchors"]
            for obj in relation["objects"]:
                relations |= get_all_relations(obj)
    return relations


def load_pc(scan_id, keep_background = False, scan_dir = 'data/referit3d/scan_data'):
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, 'pcd_with_global_alignment', '%s.pth' % scan_id))
    obj_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))

    origin_pcds = []
    batch_pcds = []
    batch_labels = []
    inst_locs = []
    obj_ids = []
    for i, obj_label in enumerate(obj_labels):
        if (not keep_background) and obj_label in ['wall', 'floor', 'ceiling']:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, 'scan: %s, obj %d' % (scan_id, i)
        obj_pcd = pcds[mask]
        # obj_color = colors[mask]
        # origin_pcds.append(np.concatenate([obj_pcd, obj_color], 1))

        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        # height_array = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()

        # # normalize
        # obj_pcd = obj_pcd - obj_pcd.mean(0)
        # max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
        # if max_dist < 1e-6:     # take care of tiny point-clouds, i.e., padding
        #     max_dist = 1
        # obj_pcd = obj_pcd / max_dist
        # obj_color = obj_color / 127.5 - 1

        # # sample points
        # pcd_idxs = np.random.choice(len(obj_pcd), size=2048, replace=(len(obj_pcd) < 2048))
        # obj_pcd = obj_pcd[pcd_idxs]
        # obj_color = obj_color[pcd_idxs]
        # obj_height = height_array[pcd_idxs]

        # batch_pcds.append(np.concatenate([obj_pcd, obj_height, obj_color], 1))
        batch_labels.append(obj_label)
        obj_ids.append(i)

    # batch_pcds = torch.from_numpy(np.stack(batch_pcds, 0))
    batch_pcds = None
    center = (pcds.max(0) + pcds.min(0)) / 2

    return batch_labels, obj_ids, inst_locs, center, batch_pcds


ids_list = ['scene0568_00', 'scene0304_00', 'scene0488_00', 'scene0412_00', 'scene0217_00', 'scene0019_00', 'scene0414_00', 'scene0575_00', 'scene0426_00', 'scene0549_00', 'scene0578_00', 'scene0665_00', 'scene0050_00', 'scene0257_00', 'scene0025_00', 'scene0583_00', 'scene0701_00', 'scene0580_00', 'scene0565_00', 'scene0169_00', 'scene0655_00', 'scene0063_00', 'scene0221_00', 'scene0591_00', 'scene0678_00', 'scene0462_00', 'scene0427_00', 'scene0595_00', 'scene0193_00', 'scene0164_00', 'scene0598_00', 'scene0599_00', 'scene0328_00', 'scene0300_00', 'scene0354_00', 'scene0458_00', 'scene0423_00', 'scene0307_00', 'scene0606_00', 'scene0432_00', 'scene0608_00', 'scene0651_00', 'scene0430_00', 'scene0689_00', 'scene0357_00', 'scene0574_00', 'scene0329_00', 'scene0153_00', 'scene0616_00', 'scene0671_00', 'scene0618_00', 'scene0382_00', 'scene0490_00', 'scene0621_00', 'scene0607_00', 'scene0149_00', 'scene0695_00', 'scene0389_00', 'scene0377_00', 'scene0342_00', 'scene0139_00', 'scene0629_00', 'scene0496_00', 'scene0633_00', 'scene0518_00', 'scene0652_00', 'scene0406_00', 'scene0144_00', 'scene0494_00', 'scene0278_00', 'scene0316_00', 'scene0609_00', 'scene0084_00', 'scene0696_00', 'scene0351_00', 'scene0643_00', 'scene0644_00', 'scene0645_00', 'scene0081_00', 'scene0647_00', 'scene0535_00', 'scene0353_00', 'scene0559_00', 'scene0593_00', 'scene0246_00', 'scene0653_00', 'scene0064_00', 'scene0356_00', 'scene0030_00', 'scene0222_00', 'scene0338_00', 'scene0378_00', 'scene0660_00', 'scene0553_00', 'scene0527_00', 'scene0663_00', 'scene0664_00', 'scene0334_00', 'scene0046_00', 'scene0203_00', 'scene0088_00', 'scene0086_00', 'scene0670_00', 'scene0256_00', 'scene0249_00', 'scene0441_00', 'scene0658_00', 'scene0704_00', 'scene0187_00', 'scene0131_00', 'scene0207_00', 'scene0461_00', 'scene0011_00', 'scene0343_00', 'scene0251_00', 'scene0077_00', 'scene0684_00', 'scene0550_00', 'scene0686_00', 'scene0208_00', 'scene0500_00', 'scene0552_00', 'scene0648_00', 'scene0435_00', 'scene0690_00', 'scene0693_00', 'scene0700_00', 'scene0699_00', 'scene0231_00', 'scene0697_00', 'scene0474_00', 'scene0355_00', 'scene0146_00', 'scene0196_00', 'scene0702_00', 'scene0314_00', 'scene0277_00', 'scene0095_00', 'scene0015_00', 'scene0100_00', 'scene0558_00', 'scene0685_00']
scan_id_to_idx = {
    s: i for i, s in enumerate(ids_list)
}

SCANNET_CLASS_LABELS_200 = [
    "chair",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
    "wall",
    "floor"
]
