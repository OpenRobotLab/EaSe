import torch

DEVICE = torch.device('cuda')

class AgainstTheWall:
    def __init__(
        self, 
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor
    ) -> None:
        self.object_locations = object_locations.to(DEVICE)
        self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self.param = None
        self._init_params()

    def _init_params(self) -> None:
        sizes = self.object_locations[:, 3:]
        conditions = (torch.max(sizes, dim=1).values > 10.0) & (torch.min(sizes[:, :2], dim=1).values < 0.5)
        self.walls = torch.where(conditions)[0]

    def forward(self) -> torch.Tensor:
        num_objects = self.object_locations.shape[0]
        against_wall_metric = torch.zeros(num_objects, device=DEVICE)

        if len(self.walls) == 0:
            return against_wall_metric

        obj_centers = self.object_locations[:, :3]
        obj_sizes = self.object_locations[:, 3:]
        obj_mins = obj_centers - obj_sizes / 2
        obj_maxs = obj_centers + obj_sizes / 2

        wall_centers = self.object_locations[self.walls, :3]
        wall_sizes = self.object_locations[self.walls, 3:]
        wall_mins = wall_centers - wall_sizes / 2
        wall_maxs = wall_centers + wall_sizes / 2

        for obj_index in range(num_objects):
            obj_min = obj_mins[obj_index]
            obj_max = obj_maxs[obj_index]
            obj_center_z = obj_centers[obj_index, 2]

            overlaps_x = torch.clamp(torch.min(obj_max[0], wall_maxs[:, 0]) - torch.max(obj_min[0], wall_mins[:, 0]), min=0)
            overlaps_y = torch.clamp(torch.min(obj_max[1], wall_maxs[:, 1]) - torch.max(obj_min[1], wall_mins[:, 1]), min=0)
            valid_overlaps = (overlaps_x > 0) & (overlaps_y > 0)

            if valid_overlaps.any():
                wall_center_z = wall_centers[valid_overlaps, 2]
                overlaps_x = overlaps_x[valid_overlaps]
                overlaps_y = overlaps_y[valid_overlaps]
                vertical_alignment = 1 / (1 + torch.abs(obj_center_z - wall_center_z))
                contact_areas = overlaps_x * overlaps_y * vertical_alignment
                max_contact_area = contact_areas.max()
                against_wall_metric[obj_index] = max_contact_area

        return against_wall_metric

class Tall:
    def __init__(
        self, 
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
            scene_point_cloud: torch.Tensor, shape (M, 3), M is the number of points in the scene point cloud.
        """
        self.object_locations = object_locations.to(DEVICE)
        # self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self.param = None
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Tall` relation and initialize `self.param`. Specifically, the height of each object.
        """
        # Extracting the height (z-size) of each object
        self.param = self.object_locations[:, 5]  # The sixth column is the height
    
    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, ), where element i is the metric value of the `Tall` relation of object i.
        """
        # The metric for "tall" can be taken as the relative height of the object compared to other objects
        total_height = torch.sum(self.param)
        tall_metric = self.param / total_height
        return tall_metric

class Low:
    def __init__(
        self, 
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
            scene_point_cloud: torch.Tensor, shape (M, 3), M is the number of points in the scene point cloud.
        """
        self.object_locations = object_locations.to(DEVICE)
        # self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self.param = None
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Low` relation and initialize `self.param`. 
        Specifically, the inverse height of each object, indicating how low it is compared to others.
        """
        # Extracting the height (z-size) of each object and finding their base (bottom) position
        # The bottom position is calculated as center z-coordinate minus half of the height
        base_positions = self.object_locations[:, 2] - (self.object_locations[:, 5] / 2)
        # Inverting heights to consider low as a relative metric
        max_height = torch.max(base_positions)
        self.param = max_height - base_positions
    
    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, ), where element i is the metric value of the `Low` relation of object i.
        """
        # The metric for "low" can be taken as how close the base of the object is to the lowest base among all objects
        total_inverse_height = torch.sum(self.param)
        low_metric = self.param / total_inverse_height
        return low_metric

class OnTheFloor:
    def __init__(
        self, 
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
            scene_point_cloud: torch.Tensor, shape (M, 3), M is the number of points in the scene point cloud.
        """
        self.object_locations = object_locations.to(DEVICE)
        self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Initialize necessary parameters for `OnTheFloor` relation.
        Specifically, compute the bottom z-coordinate of each object.
        """
        # Extract the z-center and height (size along z-axis) of each object
        z_centers = self.object_locations[:, 2]
        heights = self.object_locations[:, 5]
        
        # Calculate the bottom z-coordinate for each object using vectorized operations
        self.bottom_z = z_centers - (heights * 0.5)
        
        # Determine the lowest point in the scene point cloud to represent the floor
        self.floor_z = self.scene_point_cloud[:, 2].min()

    def forward(self) -> torch.Tensor:
        """
        Returns a tensor of shape (N, ), where element i is the metric value of the `OnTheFloor` relation of object i.
        """
        # Calculate the proximity to the floor for each object
        # Use an exponential decay to calculate a continuous metric, higher value means closer to the floor
        proximity_to_floor = torch.exp(-(self.bottom_z - self.floor_z).abs())
        
        return proximity_to_floor

class Small:
    def __init__(
        self, 
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
            scene_point_cloud: torch.Tensor, shape (M, 3), M is the number of points in the scene point cloud.
        """
        self.object_locations = object_locations.to(DEVICE)
        # self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self.volumes = None
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Small` relation and initialize `self.param`. Specifically, volume of each object.
        """
        # Calculate volumes for each object based on their size
        sizes = self.object_locations[:, 3:6]  # Get the size columns
        self.volumes = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]  # Volume = width * height * depth
    
    def forward(self) -> torch.Tensor:
        """
        return a tensor of shape (N, ), where element i is the metric value of the `Small` relation of object i.
        """
        # The probability of being small is inversely related to the volume
        inverse_volumes = 1.0 / (self.volumes + 1e-8)  # Add small value to avoid division by zero
        max_inverse_volume = torch.max(inverse_volumes)

        # Normalize to get metric values in a range that indicates likelihood of 'smallness'
        metric_values = inverse_volumes / max_inverse_volume

        return metric_values

class Large:
    def __init__(
        self, 
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
            scene_point_cloud: torch.Tensor, shape (M, 3), M is the number of points in the scene point cloud.
        """
        self.object_locations = object_locations.to(DEVICE)
        # self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self.volumes = None
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Large` relation and initialize `self.param`. Specifically, volume of each object.
        """
        # Calculate volumes for each object based on their size
        sizes = self.object_locations[:, 3:6]  # Get the size columns
        self.volumes = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]  # Volume = width * height * depth
    
    def forward(self) -> torch.Tensor:
        """
        return a tensor of shape (N, ), where element i is the metric value of the `Large` relation of object i.
        """
        # The probability of being large is directly related to the volume
        max_volume = torch.max(self.volumes)

        # Normalize to get metric values in a range that indicates likelihood of 'largeness'
        metric_values = self.volumes / max_volume

        return metric_values

    

class AtTheCorner:
    def __init__(self, object_locations: torch.Tensor, scene_point_cloud: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object, the last three columns are the size of the object.
            scene_point_cloud: torch.Tensor, shape (M, 3), M is the number of points in the scene point cloud.
        """
        self.object_locations = object_locations.to(DEVICE)
        self.scene_point_cloud = scene_point_cloud.to(DEVICE)
        self._init_params()
    
    def _init_params(self) -> None:
        """
        Compute necessary parameters about `AtCorner` relation and initialize `self.param`.
        Specifically, calculating the boundaries of the scene.
        """
        # Calculate scene boundaries
        self.scene_min = self.scene_point_cloud.min(dim=0)[0]
        self.scene_max = self.scene_point_cloud.max(dim=0)[0]
        self.scene_size = self.scene_max - self.scene_min
        
    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, ), where element i is the metric value of the `AtCorner` relation of object i.
        """
        # Calculate object boundaries
        object_centers = self.object_locations[:, :3]
        size_half = self.object_locations[:, 3:] / 2
        object_min_bounds = object_centers - size_half
        object_max_bounds = object_centers + size_half
        
        # Calculate distances from each object's min boundary to the corners in x-y plane
        dist_x_min_normalized = torch.minimum(
            (object_min_bounds[:, 0] - self.scene_min[0]),
            (self.scene_max[0] - object_max_bounds[:, 0])
        ) / self.scene_size[0]
        
        dist_y_min_normalized = torch.minimum(
            (object_min_bounds[:, 1] - self.scene_min[1]),
            (self.scene_max[1] - object_max_bounds[:, 1])
        ) / self.scene_size[1]

        # Compute corner proximity by emphasizing the joint proximity to both x and y edges
        corner_proximity = torch.exp(-4 * (dist_x_min_normalized + dist_y_min_normalized))  # Simultaneous closeness to both axes

        return corner_proximity