import torch
DEVICE = torch.device('cuda')

class Above:
    def __init__(
        self, 
        object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()

    def _init_params(self) -> None:
        """
        Computing necessary parameters about `OnTopOf` relation and initialize `self.param`.
        """
        self.centers = self.object_locations[:, :3]
        self.sizes = self.object_locations[:, 3:]
        self.bottoms = self.centers - self.sizes / 2
        self.tops = self.centers + self.sizes / 2

    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, N), where element (i, j) is the metric value of the `OnTopOf` relation between object i and object j.
        """
        N = self.object_locations.shape[0]

        # Vectorized operations
        vertical_gaps = self.bottoms[:, None, 2] - self.tops[None, :, 2]
        vertical_metrics = torch.exp(-torch.abs(vertical_gaps).clamp(min=0.01))
        
        overlap_x = torch.max(
            torch.tensor(0.0, device=DEVICE), 
            torch.min(self.tops[:, None, 0], self.tops[None, :, 0]) - torch.max(self.bottoms[:, None, 0], self.bottoms[None, :, 0])
        )
        overlap_y = torch.max(
            torch.tensor(0.0, device=DEVICE), 
            torch.min(self.tops[:, None, 1], self.tops[None, :, 1]) - torch.max(self.bottoms[:, None, 1], self.bottoms[None, :, 1])
        )
        overlap_area = overlap_x * overlap_y
        
        areas = self.sizes[:, 0] * self.sizes[:, 1]
        union_area = areas[:, None] + areas[None, :] - overlap_area
        horizontal_metrics = overlap_area / union_area

        # Combine metrics for the final on-top metric
        on_top_metric = vertical_metrics * horizontal_metrics

        # Set diagonal elements to zero because an object cannot be on top of itself
        on_top_metric.fill_diagonal_(0)
        
        return on_top_metric

class Below:
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
        Computing necessary parameters about `Below` relation and initialize required parameters.
        """
        self.centers = self.object_locations[:, :3]
        self.sizes = self.object_locations[:, 3:]
        self.bottoms = self.centers - self.sizes / 2
        self.tops = self.centers + self.sizes / 2

    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, N), where element (i, j) is the metric value of the `Below` relation between object i and object j.
        """
        N = self.object_locations.shape[0]

        # Compute vertical distance with reverse logic
        vertical_distances = self.tops[:, None, 2] - self.bottoms[None, :, 2]

        # Apply vertical metric, capturing how much below object i is relative to j
        vertical_metric = torch.exp(-vertical_distances.clamp(min=0.0))

        # Compute horizontal distances for proximity
        horizontal_distances = torch.abs(self.centers[:, None, :2] - self.centers[None, :, :2])

        # Calculate size sensitivities
        size_ratios = (self.sizes[:, None, :2] + self.sizes[None, :, :2]) / 2

        # Horizontal proximity metric
        horizontal_metric = torch.exp(-(horizontal_distances / size_ratios).sum(dim=-1))

        # Avoid self-comparison by setting diagonal to zero
        vertical_metric.fill_diagonal_(0)
        horizontal_metric.fill_diagonal_(0)

        # Final below metric
        below_metric = vertical_metric * horizontal_metric

        return below_metric