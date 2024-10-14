#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_arc_data(grid_size=(20, 20), num_shapes=3):
    grid = np.zeros(grid_size, dtype=int)
    colors = [1, 2, 3, 4, 5] 
    for _ in range(num_shapes):
        color = np.random.choice(colors)
        shape_type = np.random.choice(['square', 'rectangle', 'line'])
        x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
        if shape_type == 'square':
            size = np.random.randint(2, 5)
            grid[x:x+size, y:y+size] = color
        elif shape_type == 'rectangle':
            width = np.random.randint(2, 5)
            height = np.random.randint(2, 5)
            grid[x:x+width, y:y+height] = color
        elif shape_type == 'line':
            length = np.random.randint(2, grid_size[0])  # Adjust for boundaries
            orientation = np.random.choice(['horizontal', 'vertical'])
            if orientation == 'horizontal':
                grid[x, y:min(y+length, grid_size[1])] = color # boundary check
            else:
                grid[x:min(x + length, grid_size[0]), y] = color # boundary check
    return grid

def visualize_grid(grid):
    color_map = {
        0: [0, 0, 0],       
        1: [255, 0, 0],     
        2: [0, 255, 0],      
        3: [0, 0, 255],      
        4: [255, 255, 0],    
        5: [255, 165, 0],  
    }
    rgb_grid = np.zeros((*grid.shape, 3), dtype=np.uint8)
    for color_index, color_value in color_map.items():
        rgb_grid[grid == color_index] = color_value
    return rgb_grid

def apply_radiance_cascades(rgb_grid, iterations=5, decay=0.9):
    light_map = np.zeros_like(rgb_grid, dtype=float)
    initial_light = rgb_grid.astype(float) / 255.0

    for i in range(iterations):
        spread_light = gaussian_filter(initial_light, sigma=1)
        initial_light = spread_light * decay
        light_map += initial_light

    max_light = np.max(light_map)
    if max_light > 0:
        light_map /= max_light

    enhanced_grid = rgb_grid.astype(float) * light_map
    enhanced_grid = np.clip(enhanced_grid, 0, 255).astype(np.uint8)
    return enhanced_grid

def main():
    plt.ioff() #Added this line
    grid = generate_arc_data()
    rgb_grid = visualize_grid(grid)
    enhanced_grid = apply_radiance_cascades(rgb_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_grid)
    axes[0].set_title('Original Grid')
    axes[0].axis('off')

    axes[1].imshow(enhanced_grid)
    axes[1].set_title('After Radiance Cascades')
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
