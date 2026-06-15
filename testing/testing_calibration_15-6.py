# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import os

# IMAGE_DIR = r"D:\Saxion\Junior\Internship - Nano Lab\Work\Images\Sweeping_Images_10-6\capturesCameraNew2\capturesCameraNew2"

# print("Loading images...")
# stack = []
# for i in range(1920):
#     fname = os.path.join(IMAGE_DIR, f"Capture_sweep_{i:04d}.bmp")
#     arr   = np.array(Image.open(fname).convert('L'), dtype=np.float32)
#     stack.append(arr)
#     if i % 100 == 0:
#         print(f"  {i}/1920")

# stack = np.stack(stack, axis=0)   # shape (1920, H, W)


# std_map = stack.std(axis=0)

# col_means = stack.mean(axis=(1,2))   # shape (1920,)

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# axes[0].imshow(std_map, cmap='hot')
# axes[0].set_title('Pixel std across all 1920 frames\nbright = sensitive to sweep boundary')
# axes[0].axis('off')

# axes[1].plot(col_means)
# axes[1].set_xlabel('SLM column position (sweep)')
# axes[1].set_ylabel('Mean camera intensity')
# axes[1].set_title('Mean intensity vs sweep position\n(should show a step when boundary crosses beam)')

# plt.tight_layout()
# plt.savefig('sweep_sensitivity_map.png', dpi=150)
# plt.show()



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

IMAGE_DIR = r"D:\Saxion\Junior\Internship - Nano Lab\Work\Images\Sweeping_Images_10-6\capturesCameraNew2\capturesCameraNew2"

print("Computing statistics iteratively (memory-efficient)...")

first_img = np.array(Image.open(os.path.join(IMAGE_DIR, f"Capture_sweep_{0:04d}.bmp")).convert('L'), dtype=np.float32)
H, W = first_img.shape

mean_map = np.zeros((H, W), dtype=np.float32)
M2_map = np.zeros((H, W), dtype=np.float32)  # sum of squared differences
col_means = np.zeros(1920, dtype=np.float32)

for i in range(1920):
    fname = os.path.join(IMAGE_DIR, f"Capture_sweep_{i:04d}.bmp")
    arr = np.array(Image.open(fname).convert('L'), dtype=np.float32)
    
    delta = arr - mean_map
    mean_map += delta / (i + 1)
    delta2 = arr - mean_map
    M2_map += delta * delta2
    
    # Store column mean
    col_means[i] = arr.mean()
    
    if i % 100 == 0:
        print(f"  {i}/1920")


std_map = np.sqrt(M2_map / 1920)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].imshow(std_map, cmap='hot')
axes[0].set_title('Pixel std across all 1920 frames\nbright = sensitive to sweep boundary')
axes[0].axis('off')

axes[1].plot(col_means)
axes[1].set_xlabel('SLM column position (sweep)')
axes[1].set_ylabel('Mean camera intensity')
axes[1].set_title('Mean intensity vs sweep position\n(should show a step when boundary crosses beam)')

plt.tight_layout()
plt.savefig('sweep_sensitivity_map.png', dpi=150)
plt.show()
