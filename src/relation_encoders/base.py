import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel
from pathlib import Path

from src.util.label import CLASS_LABELS_200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16', model_max_length=512, use_fast=True)

clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').cuda()
clip.share_memory()

class BaseConcept:
    def __init__(self, scan_data: dict, label_type=None) -> None:
        self.scan_data = scan_data
        self._init_params(label_type)
    
    def _init_params(self) -> None:
        raise NotImplementedError
    
    def forward(self) -> torch.Tensor:
        raise NotImplementedError

class CategoryConcept(BaseConcept):
    def _init_params(self, label_type) -> None:
        self.scan_id = self.scan_data['scan_id']
        if label_type == "gt":
            pred_class_list = self.scan_data['inst_labels']
        else:
            self.obj_embeds = self.scan_data['obj_embeds']
            self.class_name_list = list(CLASS_LABELS_200)
            self.class_name_list.remove('wall')
            self.class_name_list.remove('floor')
            self.class_name_list.remove('ceiling')

            self.class_name_tokens = tokenizer([f'a {class_name} in a scene.' for class_name in self.class_name_list],
                                                    padding=True,
                                                    return_tensors='pt')
            for name in self.class_name_tokens.data:
                self.class_name_tokens.data[name] = self.class_name_tokens.data[name].cuda()
            
            label_lang_infos = clip.get_text_features(**self.class_name_tokens)
            del self.class_name_tokens
            label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)
            class_logits_3d = torch.matmul(label_lang_infos, self.obj_embeds.t().cuda())     # * logit_scale
            obj_cls = class_logits_3d.argmax(dim=0)
            pred_class_list = [self.class_name_list[idx] for idx in obj_cls]
        new_class_list = pred_class_list
        new_class_name_tokens = tokenizer([f'a {class_name} in a scene.' for class_name in new_class_list],
                                               padding=True,
                                               return_tensors='pt')
        self.pred_class_list = pred_class_list
        self.obj_ids = self.scan_data['obj_ids']
        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].cuda()
        
        label_lang_infos = clip.get_text_features(**new_class_name_tokens)
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)
        
    @torch.inference_mode()
    def forward(self, category: str, color: str = None, shape: str = None) -> torch.Tensor:
        query_name_tokens = tokenizer([f'a {category} in a scene.'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()
        query_lang_infos = clip.get_text_features(**query_name_tokens)
        query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True) # (768, )
        text_cls = torch.matmul(query_lang_infos, self.label_lang_infos.t()).squeeze() # (1, 768) * (768, N) -> (N, )
        text_cls = softmax(100 * text_cls, dim=0)
        return text_cls.to(DEVICE)

class ScanReferCategoryConcept(BaseConcept):
    def _init_params(self, label_type) -> None:
        self.scan_id = self.scan_data['scan_id']
        pred_class_list = self.scan_data['pred_labels']
        new_class_list = pred_class_list
        new_class_name_tokens = tokenizer([f'a {class_name} in a scene.' for class_name in new_class_list],
                                               padding=True,
                                               return_tensors='pt')
        self.pred_class_list = pred_class_list
        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].cuda()
        label_lang_infos = clip.get_text_features(**new_class_name_tokens)
        del new_class_name_tokens
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)
    
    def forward(self, category: str, color: str = None, shape: str = None) -> torch.Tensor:
        query_name_tokens = tokenizer([f'a {category} in a scene.'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()
        with torch.no_grad():
            query_lang_infos = clip.get_text_features(**query_name_tokens)
            query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True) # (768, )
            text_cls = torch.matmul(query_lang_infos, self.label_lang_infos.t()).squeeze() # (1, 768) * (768, N) -> (N, )
            text_cls = softmax(100 * text_cls, dim=0)     
            
        return text_cls.to(DEVICE)

class Near:
    def __init__(
        self, 
        object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()

    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Near` relation and initialize `self.param`.
        """
        # Based on average size to calculate a baseline distance
        sizes = self.object_locations[:, 3:]
        self.avg_size_norm = sizes.mean(dim=0).norm().to(DEVICE)

    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, N), where element (i, j) is the metric value of the `Near` relation between object i and object j.
        """
        centers = self.object_locations[:, :3]
        sizes = self.object_locations[:, 3:]
        
        # Calculate the pairwise distances between object centers in a vectorized manner
        diff = centers.unsqueeze(1) - centers.unsqueeze(0)
        distances = diff.norm(dim=2)
        
        # Calculate the "nearness" metric based on distances and average size norm
        nearness_metric = torch.exp(-distances / (self.avg_size_norm + 1e-6))
        
        # Set diagonal to zero since an object cannot be near itself
        nearness_metric.fill_diagonal_(0)
        
        return nearness_metric.to(DEVICE)

class Far:
    def __init__(
        self, 
        object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()

    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Far` relation and initialize `self.param`.
        """
        # Based on average size to calculate a baseline distance
        sizes = self.object_locations[:, 3:]
        self.avg_size_norm = sizes.mean(dim=0).norm().to(DEVICE)

    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, N), where element (i, j) is the metric value of the `Far` relation between object i and object j.
        """
        centers = self.object_locations[:, :3]
        sizes = self.object_locations[:, 3:]
        
        # Calculate the pairwise distances between object centers in a vectorized manner
        diff = centers.unsqueeze(1) - centers.unsqueeze(0)
        distances = diff.norm(dim=2)
        
        # Calculate the "farness" metric based on distances and average size norm
        farness_metric = 1.0 - torch.exp(-distances / (self.avg_size_norm + 1e-6))
        
        # Set diagonal to zero since an object cannot be far from itself
        farness_metric.fill_diagonal_(0)
        
        return farness_metric.to(DEVICE)
    