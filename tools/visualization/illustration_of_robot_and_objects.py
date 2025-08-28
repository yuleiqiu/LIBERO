import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axes
fig, ax = plt.subplots()

# Original Coordinates
robot_coords = (-0.6, 0)
basket_coords = (-0.01, 0.30)

# Rotate coordinates clockwise by 90 degrees (x' = y, y' = -x)
# This makes the original X-axis point downwards.
rotated_robot_coords = (robot_coords[1], -robot_coords[0])
rotated_basket_coords = (basket_coords[1], -basket_coords[0])

# Plot the Robot and Basket
ax.plot(rotated_robot_coords[0], rotated_robot_coords[1], 'rs', markersize=10, label='Robot')
ax.plot(rotated_basket_coords[0], rotated_basket_coords[1], 'bo', markersize=10, label='Basket')

# Add text labels for the Robot and Basket
ax.text(rotated_robot_coords[0], rotated_robot_coords[1] + 0.05, 'Robot', ha='center')
ax.text(rotated_basket_coords[0], rotated_basket_coords[1] + 0.05, 'Basket', ha='center')

# --- Add the Rectangular Region ---
# Original rectangle vertices
# The data -0.4 -0.4 0.1 0.1 defines the min/max x and y values.
# x_min, y_min, x_max, y_max = -0.4, -0.4, 0.1, 0.1
rect_x_min = -0.4
rect_y_min = -0.4
rect_width_orig = 0.1 - (-0.4)
rect_height_orig = 0.1 - (-0.4)

# We need to transform the rectangle's position and dimensions.
# The bottom-left corner in the original space is (rect_x_min, rect_y_min)
# Transform this corner: (y, -x)
rotated_corner = (rect_y_min, -rect_x_min)

# After a 90-degree clockwise rotation, the original width becomes the new height,
# and the original height becomes the new width.
new_rect_width = rect_height_orig
new_rect_height = rect_width_orig
# However, the drawing anchor point also changes. The simplest way is to find the new
# bottom-left corner of the transformed rectangle.
# Transformed vertices: (-0.4, 0.4), (0.1, 0.4), (-0.4, -0.1), (0.1, -0.1)
new_bottom_left_x = -0.4
new_bottom_left_y = -0.1

# Create the rectangle patch
rect = patches.Rectangle(
    (new_bottom_left_x, new_bottom_left_y),
    new_rect_width,
    new_rect_height,
    linewidth=1,
    edgecolor='g',
    facecolor='g',
    alpha=0.3, # transparency
    label='Region'
)

# Add the patch to the Axes
ax.add_patch(rect)
# --- End of Adding Rectangle ---


# Set plot limits and labels
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.25, 0.8)
ax.set_xlabel('Original Y-Coordinate')
ax.set_ylabel('Original X-Coordinate')
ax.set_title('Top-Down View (X-axis Downward)')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.legend()

# Save the figure
plt.savefig('rotated_schematic_diagram_with_rectangle.png')