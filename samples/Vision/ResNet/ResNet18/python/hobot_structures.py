# hobot_structures.py
import ctypes

class hbDNNQuantiType(ctypes.c_int):
    """
    Enumeration for DNN quantization type.
    """
    NONE = 0
    SCALE = 1

class hbSysMem_t(ctypes.Structure):
    """
    Represents system memory buffer.
    """
    _fields_ = [
        ("phyAddr", ctypes.c_void_p),       # Physical address
        ("virAddr", ctypes.c_void_p),       # Virtual address
        ("memSize", ctypes.c_int)           # Memory size
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    """
    Represents quantization scale parameters.
    """
    _fields_ = [
        ("scaleLen", ctypes.c_int),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int),
        ("zeroPointData", ctypes.POINTER(ctypes.c_int32)) # Pointer to int32
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    """
    Represents the shape of a DNN tensor.
    """
    _fields_ = [
        ("dimensionSize", ctypes.c_int * 8), # Supports up to 8 dimensions
        ("numDimensions", ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    """
    Properties of a DNN tensor for the C library.
    """
    _fields_ = [
        ("validShape", hbDNNTensorShape_t),
        ("tensorType", ctypes.c_int),       # Tensor data type (e.g., float, int8)
        ("scale", hbDNNQuantiScale_t),      # Quantization scale info
        ("quantiType", hbDNNQuantiType),    # Quantization type (NONE or SCALE)
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize", ctypes.c_longlong), # int64_t
        ("stride", ctypes.c_longlong * 8)       # int64_t array
    ]

class hbDNNTensor_t(ctypes.Structure):
    """
    Represents a DNN tensor for the C library.
    """
    _fields_ = [
        ("sysMem", hbSysMem_t),
        ("properties", hbDNNTensorProperties_t)
    ]

class ClassificationPostProcessInfo_t(ctypes.Structure):
    """
    Parameters for the classification post-processing C function.
    """
    _fields_ = [
        ("height", ctypes.c_int),           # Model input height
        ("width", ctypes.c_int),            # Model input width
        ("ori_height", ctypes.c_int),       # Original image height
        ("ori_width", ctypes.c_int),        # Original image width
        ("score_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("nms_top_k", ctypes.c_int),
        ("is_pad_resize", ctypes.c_int),    # Boolean flag (0 or 1)
        ("use_softmax", ctypes.c_bool)      # Boolean flag
    ]