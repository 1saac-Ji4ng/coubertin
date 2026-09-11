import numpy as np
import matplotlib.pyplot as plt

WORLD_IMAGE_PATH = "Your path"

UNITS_PER_PIXEL = None

world_image = plt.imread(WORLD_IMAGE_PATH)

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(world_image)
ax.set_title(
    "Click: Line 1 endpoint 1, Line 1 endpoint 2, "
    "Line 2 endpoint 1, Line 2 endpoint 2"
)

print("click four points in world.png")

pixel_points = np.array(
    plt.ginput(4, timeout=-1)
)

world_points = pixel_points * UNITS_PER_PIXEL

# choose two lines in the picture
ax.plot(
    pixel_points[:2, 0],
    pixel_points[:2, 1],
    "b-",
    linewidth=3
)

ax.plot(
    pixel_points[2:, 0],
    pixel_points[2:, 1],
    "r-",
    linewidth=3
)

fig.canvas.draw()

print("\nPixel coordinates:")
print(pixel_points)

print("\nWorld coordinates:")
print(world_points)

print("\nCopy into the notebook:")
print(
    f"line1_p1 = moving.Point"
    f"({world_points[0, 0]:.4f}, {world_points[0, 1]:.4f})"
)
print(
    f"line1_p2 = moving.Point"
    f"({world_points[1, 0]:.4f}, {world_points[1, 1]:.4f})"
)
print(
    f"line2_p1 = moving.Point"
    f"({world_points[2, 0]:.4f}, {world_points[2, 1]:.4f})"
)
print(
    f"line2_p2 = moving.Point"
    f"({world_points[3, 0]:.4f}, {world_points[3, 1]:.4f})"
)

plt.show()