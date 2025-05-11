import numpy as np
import matplotlib.pyplot as plt

class OccupancyGridMap:
    """
    A class to represent a 2D occupancy grid map.
    """

    def __init__(
        self,
        width_m,
        height_m,
        resolution,
        origin_x_m=0.0,
        origin_y_m=0.0,
        initial_prob=0.5,
    ):
        """
        Initializes the occupancy grid map.

        Args:
            width_m (float): The width of the map area in meters.
            height_m (float): The height of the map area in meters.
            resolution (float): The size of each grid cell in meters.
            origin_x_m (float): The x-coordinate of the map's origin (bottom-left corner)
                                in the world frame (meters). Defaults to 0.0.
            origin_y_m (float): The y-coordinate of the map's origin (bottom-left corner)
                                in the world frame (meters). Defaults to 0.0.
            initial_prob (float): The initial occupancy probability for all cells
                                  (0.0 to 1.0). Defaults to 0.5 (unknown).
        """
        self.resolution = resolution
        self.origin_x = origin_x_m
        self.origin_y = origin_y_m

        # Calculate grid dimensions in cells
        self.width_cells = int(np.ceil(width_m / resolution))
        self.height_cells = int(np.ceil(height_m / resolution))

        # Initialize the grid with the initial probability
        # Using float32 for memory efficiency
        self.grid = np.full(
            (self.height_cells, self.width_cells),
            initial_prob,
            dtype=np.float32,
        )

        print(
            f"Initialized grid: {self.width_cells}x{self.height_cells} cells, "
            f"resolution: {self.resolution} m/cell"
        )
        print(
            f"Map world bounds: "
            f"x=[{self.origin_x}, {self.origin_x + self.width_cells * self.resolution}], "
            f"y=[{self.origin_y}, {self.origin_y + self.height_cells * self.resolution}]"
        )

    def world_to_grid(self, world_x, world_y):
        """
        Converts world coordinates (meters) to grid cell indices.

        Args:
            world_x (float): X-coordinate in the world frame (meters).
            world_y (float): Y-coordinate in the world frame (meters).

        Returns:
            tuple (int, int) or None: (row, col) indices of the cell, or None if outside map.
                                      Row corresponds to y, Col corresponds to x.
        """
        grid_col = int((world_x - self.origin_x) / self.resolution)
        grid_row = int((world_y - self.origin_y) / self.resolution)

        # Check if the coordinates are within the grid boundaries
        if (
            0 <= grid_row < self.height_cells
            and 0 <= grid_col < self.width_cells
        ):
            return (grid_row, grid_col)
        else:
            # print(f"Warning: World coordinates ({world_x}, {world_y}) are outside the map.")
            return None

    def grid_to_world(self, grid_row, grid_col):
        """
        Converts grid cell indices to world coordinates (meters) at the cell center.

        Args:
            grid_row (int): Row index of the cell.
            grid_col (int): Column index of the cell.

        Returns:
            tuple (float, float) or None: (x, y) world coordinates, or None if indices are invalid.
        """
        if (
            0 <= grid_row < self.height_cells
            and 0 <= grid_col < self.width_cells
        ):
            world_x = (
                self.origin_x + (grid_col + 0.5) * self.resolution
            )  # Center of cell
            world_y = (
                self.origin_y + (grid_row + 0.5) * self.resolution
            )  # Center of cell
            return (world_x, world_y)
        else:
            # print(f"Warning: Grid indices ({grid_row}, {grid_col}) are outside the map.")
            return None

    def is_valid_grid_coords(self, grid_row, grid_col):
        """Checks if grid coordinates are within the map bounds."""
        return (
            0 <= grid_row < self.height_cells
            and 0 <= grid_col < self.width_cells
        )

    def set_probability(self, grid_row, grid_col, prob):
        """
        Sets the occupancy probability of a specific cell. Clips value to [0, 1].

        Args:
            grid_row (int): Row index of the cell.
            grid_col (int): Column index of the cell.
            prob (float): The new occupancy probability (0.0 to 1.0).
        """
        if self.is_valid_grid_coords(grid_row, grid_col):
            self.grid[grid_row, grid_col] = np.clip(prob, 0.0, 1.0)
        else:
            print(
                f"Warning: Attempted to set probability outside map at ({grid_row}, {grid_col})"
            )

    def get_probability(self, grid_row, grid_col):
        """
        Gets the occupancy probability of a specific cell.

        Args:
            grid_row (int): Row index of the cell.
            grid_col (int): Column index of the cell.

        Returns:
            float or None: The occupancy probability, or None if indices are invalid.
        """
        if self.is_valid_grid_coords(grid_row, grid_col):
            return self.grid[grid_row, grid_col]
        else:
            # print(f"Warning: Attempted to get probability outside map at ({grid_row}, {grid_col})")
            return None

    def update_cell_log_odds(self, grid_row, grid_col, log_odds_update):
        """
        Updates a cell's probability using log-odds addition.

        Args:
            grid_row (int): Row index of the cell.
            grid_col (int): Column index of the cell.
            log_odds_update (float): The log-odds value to add (from sensor model).
                                     Positive for occupied evidence, negative for free evidence.
        """
        if not self.is_valid_grid_coords(grid_row, grid_col):
            print(
                f"Warning: Attempted log-odds update outside map at ({grid_row}, {grid_col})"
            )
            return

        # Get current probability and convert to log-odds
        current_prob = self.grid[grid_row, grid_col]
        # Avoid division by zero or log(0) using epsilon
        epsilon = 1e-6
        current_prob_clipped = np.clip(current_prob, epsilon, 1.0 - epsilon)
        current_log_odds = np.log(
            current_prob_clipped / (1.0 - current_prob_clipped)
        )

        # Add the update
        new_log_odds = current_log_odds + log_odds_update

        # Clamp log-odds to prevent extreme probabilities (optional but recommended)
        # These bounds correspond to probabilities very close to 0 and 1
        log_odds_max = 5.0  # Corresponds to p ~ 0.993
        log_odds_min = -5.0 # Corresponds to p ~ 0.007
        new_log_odds_clamped = np.clip(new_log_odds, log_odds_min, log_odds_max)


        # Convert back to probability
        new_prob = 1.0 - (1.0 / (1.0 + np.exp(new_log_odds_clamped)))

        self.grid[grid_row, grid_col] = new_prob

    def visualize(self, title="Occupancy Grid Map", ax=None, show_colorbar=True):
        """
        Visualizes the occupancy grid map using Matplotlib.

        Args:
            title (str): The title for the plot.
            ax (matplotlib.axes.Axes, optional): An existing Axes object to plot on.
                                                 If None, a new figure and axes are created.
            show_colorbar (bool): Whether to display the colorbar.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Define the extent for imshow to show world coordinates
        extent = [
            self.origin_x,
            self.origin_x + self.width_cells * self.resolution,
            self.origin_y,
            self.origin_y + self.height_cells * self.resolution,
        ]

        # Use imshow to display the grid
        # origin='lower' puts (0,0) index at the bottom-left, matching typical map coordinates
        # cmap='gray_r' makes 0=white (free), 0.5=gray (unknown), 1=black (occupied)
        im = ax.imshow(
            self.grid,
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            origin="lower",
            interpolation="nearest", # Show discrete cells clearly
        )

        ax.set_title(title)
        ax.set_xlabel("X coordinate (meters)")
        ax.set_ylabel("Y coordinate (meters)")
        ax.set_aspect("equal") # Ensure cells are square if resolution is uniform

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Occupancy Probability")

        # Optionally show grid lines (can be slow for large grids)
        # ax.set_xticks(np.arange(extent[0], extent[1] + self.resolution, self.resolution), minor=True)
        # ax.set_yticks(np.arange(extent[2], extent[3] + self.resolution, self.resolution), minor=True)
        # ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        # ax.tick_params(which='minor', size=0) # Hide minor tick marks

        if ax is None: # Only call show if we created the figure
             plt.show()


# --- Example Usage ---

if __name__ == "__main__":
    # Create a 10m x 8m map with 0.1m resolution
    # Origin at (-2, -1) in the world frame
    occ_map = OccupancyGridMap(
        width_m=10.0, height_m=8.0, resolution=0.1, origin_x_m=-2.0, origin_y_m=-1.0
    )

    # --- Method 1: Setting probabilities directly ---
    print("\n--- Setting Probabilities Directly ---")
    # Mark a cell as occupied using grid coordinates
    occ_map.set_probability(grid_row=20, grid_col=30, prob=0.95) # Occupied

    # Mark a cell as free using world coordinates
    world_coords_free = (2.55, 3.05) # Center of cell (2.5, 3.0) -> grid (45, 40)
    grid_coords_free = occ_map.world_to_grid(world_coords_free[0], world_coords_free[1])
    if grid_coords_free:
        print(f"World coords {world_coords_free} correspond to grid coords {grid_coords_free}")
        occ_map.set_probability(grid_coords_free[0], grid_coords_free[1], prob=0.05) # Free
    else:
        print(f"World coords {world_coords_free} are outside the map.")

    # Get probability of a cell
    prob = occ_map.get_probability(20, 30)
    print(f"Probability at grid (20, 30): {prob:.2f}")
    prob_unknown = occ_map.get_probability(5, 5)
    print(f"Probability at grid (5, 5): {prob_unknown:.2f}") # Should be initial value (0.5)

    # Visualize the map after direct updates
    occ_map.visualize(title="Map After Direct Probability Updates")


    # --- Method 2: Updating using Log-Odds ---
    print("\n--- Updating Using Log-Odds ---")
    # Reset a cell to unknown for demonstration
    occ_map.set_probability(grid_row=50, grid_col=60, prob=0.5)

    # Simulate sensor readings for cell (50, 60)
    # Assume sensor model gives log-odds update values
    log_odds_occupied = 1.386 # Corresponds to p=0.8, P(occ|z) / (1-P(occ|z)) = 4 -> log(4)
    log_odds_free = -0.847 # Corresponds to p=0.3, P(occ|z) / (1-P(occ|z)) = 0.42 -> log(0.42)

    print(f"Initial probability at (50, 60): {occ_map.get_probability(50, 60):.2f}")

    # First update: evidence of occupation
    occ_map.update_cell_log_odds(50, 60, log_odds_occupied)
    print(f"After 1st update (occupied): {occ_map.get_probability(50, 60):.2f}")

    # Second update: more evidence of occupation
    occ_map.update_cell_log_odds(50, 60, log_odds_occupied)
    print(f"After 2nd update (occupied): {occ_map.get_probability(50, 60):.2f}")

    # Third update: evidence of free space
    occ_map.update_cell_log_odds(50, 60, log_odds_free)
    print(f"After 3rd update (free): {occ_map.get_probability(50, 60):.2f}")

    # Visualize the map after log-odds updates
    occ_map.visualize(title="Map After Log-Odds Updates")

    # Example: Draw a line of obstacles
    print("\n--- Drawing a line ---")
    for i in range(20):
        row, col = 10 + i, 5 + i*2
        if occ_map.is_valid_grid_coords(row, col):
             occ_map.update_cell_log_odds(row, col, log_odds_occupied * 2) # Stronger evidence

    occ_map.visualize(title="Map with Obstacle Line")
