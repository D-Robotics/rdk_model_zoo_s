


DEVICE_TREE_PATH = "/sys/firmware/devicetree/base/model"


class DeviceSelect:
    def __init__(self):
        try:
            with open(DEVICE_TREE_PATH, "r") as f:
                tree = f.read()
        except (FileNotFoundError, PermissionError, OSError):
            raise RuntimeError("Unsupported device: cannot read device tree.")
        if 'RDK S100P' in tree:
            self.device = 'rdks100p'
        elif 'RDK S100' in tree:
            self.device = 'rdks100'
        elif 'RDK X5' in tree:
            self.device = 'rdkx5'
        else:
            raise RuntimeError("Unsupported device: {}".format(tree))
    
    def __call__(self):
        return self.device