import numpy as np
import json
import torch

data = np.load("video0.npy")

with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "") for k, v in kinetics_classnames.items()}

from hbm_runtime import HB_HBMRuntime
inf = HB_HBMRuntime("r3d_18.hbm")

pred = inf.run(data)['r3d_18']['output']
pred = torch.tensor(pred)
topk = pred.topk(k=5).indices[0]

pred_class_names = [kinetics_id_to_classname[int(i)] for i in topk]
print("Predicted labels:", ", ".join(pred_class_names))
