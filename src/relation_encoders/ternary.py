import numpy as np
import torch

class Between:
    def __init__(self, object_locations: torch.Tensor, device='cuda') -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
            device: The device where the tensor will be allocated.
        """
        self.object_locations = object_locations.to(device)
        self.N = object_locations.shape[0]
        self.device = device

    def forward(self) -> torch.Tensor:
        # Extract centers
        centers = self.object_locations[:, :3]

        # Expand dimensions for vectorized operations
        centers_expanded_i = centers.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N, 3)
        centers_expanded_j = centers.unsqueeze(0).unsqueeze(2)  # Shape: (1, N, 1, 3)
        centers_expanded_k = centers.unsqueeze(1).unsqueeze(2)  # Shape: (N, 1, 1, 3)

        # Compute vector from object i to object j
        ij_vector = centers_expanded_j - centers_expanded_i  # Shape: (N, N, 1, 3)
        ij_distance = torch.norm(ij_vector, dim=-1, keepdim=True)  # Shape: (N, N, 1)

        # Compute vector from object i to object k
        ik_vector = centers_expanded_k - centers_expanded_i  # Shape: (N, 1, N, 3)
        
        # Compute projection of ik_vector onto ij_vector
        projection_length = torch.sum(ik_vector * ij_vector, dim=-1) / (ij_distance.squeeze(-1) + 1e-6)  # Shape: (N, N, N)
        
        # Compute perpendicular distance
        perpendicular_distance_squared = torch.sum(ik_vector**2, dim=-1) - (projection_length**2)
        perpendicular_distance = torch.sqrt(torch.clamp(perpendicular_distance_squared, min=0))
        
        # Check if k is between i and j
        in_between_mask = (projection_length >= 0) & (projection_length <= ij_distance.squeeze(-1))
        
        # Compute size influence
        size_influence = torch.exp(-perpendicular_distance)
        
        # Apply the mask
        between_metrics = size_influence * in_between_mask.float()
        
        # Set diagonal elements to zero (i == j cases)
        between_metrics[range(self.N), range(self.N), :] = 0
        between_metrics[:, range(self.N), range(self.N)] = 0
        between_metrics[range(self.N), :, range(self.N)] = 0

        return between_metrics
