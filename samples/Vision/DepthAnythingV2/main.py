from hbm_runtime import HB_HBMRuntime
inf = HB_HBMRuntime("depth_any.hbm")
import os
import numpy as np
import torch
    
pixel_values_np = np.load("pixel_values.npy")
pred_depth = inf.run(pixel_values_np)['depth_any']['predicted_depth'] # (1. 518, 686)
pred_depth = torch.tensor(pred_depth)

import torch.nn.functional as F
import numpy as np
import cv2

h = 1824
w = 2400
depth = F.interpolate(pred_depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

depth = depth.cpu().detach().numpy().astype(np.uint8)
depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

from matplotlib import pyplot as plt

plt.imshow(depth_color[:, :, ::-1])
plt.savefig("depth_color.png", dpi=300, bbox_inches='tight')