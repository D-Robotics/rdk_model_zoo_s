import libmodel_task
import onnxruntime as ort
import numpy as np
import os


class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        self.sess = ort.InferenceSession(onnxfile)
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor
        return output
    
    def __del__(self):
        print('{} unload'.format(self.onnxfile))
        
        
def main():
    inf = libmodel_task.ModelTask()
    inf.ModelInit("vfe.hbm")
    voxels = np.load("voxels0.npy")
    voxel_num_points = np.load("voxel_num_points0.npy")
    voxel_coords = np.load("voxel_coords0.npy")
    pillar_features = inf.ModelInfer([voxels, voxel_num_points, voxel_coords])
    pillar_features = np.array(pillar_features[0], dtype=np.float32).reshape(11546, 64)
    del inf
    np.save('pillar_features.npy', pillar_features)
    
    point_pillar_scatter_onnx = OrtWrapper(onnxfile="PointPillarScatter.onnx")
    scatter_inputs = {
        'pillar_features': pillar_features,
        'voxel_coords': voxel_coords,
    }
    spatial_features = point_pillar_scatter_onnx.forward(scatter_inputs)['spatial_features']
    np.save('spatial_features.npy', spatial_features)
    
    inf = libmodel_task.ModelTask()
    inf.ModelInit("backbone.hbm")
    spatial_features_2d = inf.ModelInfer([spatial_features])
    spatial_features_2d = np.array(spatial_features_2d[0], dtype=np.float32).reshape(1, 384, 248, 216)
    del inf
    np.save('spatial_features_2d.npy', spatial_features_2d)
    
    anchor_onnx = OrtWrapper(onnxfile="AnchorHeadSingle.onnx")
    anchor_inputs = {
        'spatial_features_2d': spatial_features_2d,
    }
    anchor_output = anchor_onnx.forward(anchor_inputs)
    np.save('batch_cls_preds.npy', anchor_output['batch_cls_preds'])
    np.save('batch_box_preds.npy', anchor_output['batch_box_preds'])
    
    
if __name__ == "__main__":
    main()