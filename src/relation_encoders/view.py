import torch

from src.relation_encoders.base import DEVICE

class Left:
    def __init__(self, object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
        """
        self.object_locations = object_locations.to(DEVICE)
        self.N = object_locations.shape[0]

    def forward(self) -> torch.Tensor:
        # Extract centers
        centers = self.object_locations[:, :3]
        
        # Generate relative vectors from each center to every other center
        # Shape of differences: (N, N, 3)
        differences = centers.unsqueeze(1) - centers.unsqueeze(0)
        
        # Calculate cross-product z-component using batch operations
        vectors_from_origin = centers.unsqueeze(1)  # Shape (N, 1, 3)
        cross_products_z = vectors_from_origin[..., 0] * differences[..., 1] - vectors_from_origin[..., 1] * differences[..., 0]
        
        # Determine left relations using the cross-products' z-component
        left_relations = torch.where(cross_products_z > 0, 1.0, 0.0)
        
        # Calculate distances between centers
        distances = torch.norm(differences, dim=-1)
        
        # Avoid division by zero by adding a small epsilon to distances
        epsilon = 1e-6
        size_influence = 1 / (1 + distances + epsilon)

        # Apply the size influence factor only for left relations
        left_relations *= size_influence
        
        # Mask the diagonal (self-relations) to be 0, as an object cannot be 'on the left' of itself
        left_relations.fill_diagonal_(0)

        return left_relations.to(DEVICE)

class Right:
    def __init__(self, object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
        """
        self.object_locations = object_locations.to(DEVICE)
        self.N = object_locations.shape[0]

    def forward(self) -> torch.Tensor:
        # Extract centers
        centers = self.object_locations[:, :3]
        
        # Generate relative vectors from each center to every other center
        differences = centers.unsqueeze(1) - centers.unsqueeze(0)  # Shape (N, N, 3)
        
        # Calculate cross-product z-component using batch operations
        # vectors_from_origin shape: (N, 1, 3)
        vectors_from_origin = centers.unsqueeze(1)
        
        # Calculate the z-component of the cross product for the x and y differences
        cross_products_z = vectors_from_origin[..., 0] * differences[..., 1] - vectors_from_origin[..., 1] * differences[..., 0]
        
        # Determine right relations using the cross-products' z-component
        right_relations = torch.where(cross_products_z < 0, 1.0, 0.0)
        
        # Calculate distances between centers
        distances = torch.norm(differences, dim=-1)  # Shape (N, N)
        
        # Avoid division by zero by adding a small epsilon to distances
        epsilon = 1e-6
        size_influence = 1 / (1 + distances + epsilon)
        
        # Apply the size influence factor only for right relations
        right_relations *= size_influence
        
        # Mask the diagonal (self-relations) to be 0, as an object cannot be 'on the right' of itself
        right_relations.fill_diagonal_(0)

        return right_relations.to(DEVICE)

class Front:
    def __init__(self, object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Initialize necessary parameters for calculating 'in front of' relation.
        """
        # Calculate center and size for each object
        self.centers = self.object_locations[:, :3]
        self.sizes = self.object_locations[:, 3:]
    
    def forward(self) -> torch.Tensor:
        """
        Returns a tensor of shape (N, N), where element (i, j) is the metric value
        of the 'in front of' relation between object i and object j.
        """
        num_objects = self.centers.shape[0]

        # Create a grid for pairwise comparisons
        centers_i = self.centers.unsqueeze(1)  # Shape: (N, 1, 3)
        centers_j = self.centers.unsqueeze(0)  # Shape: (1, N, 3)

        # Vector from object j to object i
        direction_vectors = centers_i - centers_j  # Shape: (N, N, 3)

        # Normalize the direction vectors
        norm_direction_vectors = direction_vectors / direction_vectors.norm(dim=2, keepdim=True)  # Shape: (N, N, 3)

        # Considering object sizes in the horizontal plane (x-y)
        size_ratios_i = self.sizes[:, :2].norm(dim=1)  # Shape: (N,)
        size_ratios_j = self.sizes[:, :2].norm(dim=1)  # Shape: (N,)
        
        size_ratios_i = size_ratios_i.unsqueeze(1)  # Shape: (N, 1)
        size_ratios_j = size_ratios_j.unsqueeze(0)  # Shape: (1, N)

        # Calculate horizontal distances
        dx = centers_i[:, :, 0] - centers_j[:, :, 0]  # Shape: (N, N)
        dy = centers_i[:, :, 1] - centers_j[:, :, 1]  # Shape: (N, N)
        horizontal_distances = torch.sqrt(dx**2 + dy**2)  # Shape: (N, N)

        # Calculate alignment score
        alignment_scores = (norm_direction_vectors[:, :, :2] * torch.stack((dx, dy), dim=2) / horizontal_distances.unsqueeze(2)).sum(dim=2)  # Shape: (N, N)
        
        # Calculate size influence
        size_influences = size_ratios_i + size_ratios_j  # Shape: (N, N)
        influence_factors = size_influences / (horizontal_distances + 1e-6)  # Shape: (N, N), avoiding division by zero

        # Final metric, scaled and thresholded
        metrics = torch.relu(alignment_scores) * influence_factors  # Shape: (N, N)

        # Ensure no self-comparison
        metrics.fill_diagonal_(0)

        return metrics
    
class Behind:
    def __init__(self, object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Initialize necessary parameters for calculating 'behind' relation.
        """
        # Calculate center and size for each object
        self.centers = self.object_locations[:, :3]
        self.sizes = self.object_locations[:, 3:]
    
    def forward(self) -> torch.Tensor:
        """
        Returns a tensor of shape (N, N), where element (i, j) is the metric value
        of the 'behind' relation between object i and object j.
        """
        num_objects = self.centers.shape[0]

        # Create a grid for pairwise comparisons
        centers_i = self.centers.unsqueeze(1)  # Shape: (N, 1, 3)
        centers_j = self.centers.unsqueeze(0)  # Shape: (1, N, 3)

        # Vector from object i to object j
        direction_vectors = centers_j - centers_i  # Shape: (N, N, 3)

        # Normalize the direction vectors
        norm_direction_vectors = direction_vectors / (direction_vectors.norm(dim=2, keepdim=True) + 1e-6)  # Shape: (N, N, 3)

        # Considering object sizes in the horizontal plane (x-y)
        size_ratios_i = self.sizes[:, :2].norm(dim=1, keepdim=True)  # Shape: (N, 1)
        size_ratios_j = self.sizes[:, :2].norm(dim=1).unsqueeze(0)  # Shape: (1, N)

        # Calculate horizontal distances
        horizontal_distances = (direction_vectors[..., :2].norm(dim=2) + 1e-6)  # Shape: (N, N)

        # Calculate alignment score for "behind"
        alignment_scores = (norm_direction_vectors[..., :2] * direction_vectors[..., :2]).sum(dim=2) / horizontal_distances  # Shape: (N, N)
        
        # Calculate size influence
        size_influences = size_ratios_i + size_ratios_j  # Shape: (N, N)
        influence_factors = size_influences / horizontal_distances  # Shape: (N, N), already safe from division by zero due to +1e-6

        # Final metric, scaled and thresholded
        metrics = torch.relu(alignment_scores) * influence_factors  # Shape: (N, N)

        # Ensure no self-comparison
        metrics.fill_diagonal_(0)

        return metrics
    