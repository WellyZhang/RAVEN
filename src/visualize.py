import numpy as np
import matplotlib.pyplot as plt

# Load your .npz file
data = np.load("./matrices/distribute_four/RAVEN_0_train.npz")

# Check what it contains
print(data.files)  # Usually contains: ['image']

# Display the matrix (should be shape (8, 160, 160))
images = data['image']
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i in range(9):
    ax = axes[i // 3, i % 3]
    if i < 8:
        ax.imshow(images[i], cmap='gray')
    else:
        ax.set_title("Choose the answer")  # The target is missing
        ax.axis("off")
    ax.axis("off")

plt.tight_layout()
plt.show()
