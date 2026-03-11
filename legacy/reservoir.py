import numpy as np

class Reservoir:

    def __init__(self, layers):

        if not isinstance(layers, list):
            layers = [layers]
        self.layers = layers

        self.n_layers = len(layers)

        if all(layer.nx == layers[0].nx for layer in layers) & all(layer.ny == layers[0].ny for layer in layers):
            self.nx = layers[0].nx
            self.ny = layers[0].ny
            self.nz_list = [layer.nz for layer in layers]
            self.nz = sum(self.nz_list)
        else:
            raise ValueError("Not all nx or ny values across all layers are equal")
        
        if all(layer.x_len == layers[0].x_len for layer in layers) & all(layer.y_len == layers[0].y_len for layer in layers):
            self.x_len, self.y_len = layers[0].x_len, layers[0].y_len
            self.X, self.Y = layers[0].X, layers[0].Y
            self.z_len_list = [layer.z_len for layer in layers]
            self.z_len = sum(self.z_len_list)
        else:
            raise ValueError("Not all x_len or y_len values across all layers are equal")

        if all(np.allclose(layers[i].zz[-1], layers[i+1].zz[0]) for i in range(self.n_layers-1)):
            # self.dip = layers[0].dip
            self.top_depth = layers[0].top_depth
            self.zz = [layer.zz for layer in layers]
        else:
            raise ValueError("Bottom of layer does not match with top of next layer")
        
        self.poro_ave = [layer.poro_ave for layer in layers]
        self.poro_std = [layer.poro_std for layer in layers]
        self.perm_ave = [layer.perm_ave for layer in layers]
        self.perm_std = [layer.perm_std for layer in layers]
        self.kzkx = [layer.kzkx for layer in layers]
        self.ntg = [layer.ntg for layer in layers]

        self.poro_mat = np.concatenate([layer.poro_mat for layer in layers],axis=2)
        self.perm_mat = np.concatenate([layer.perm_mat for layer in layers],axis=2)
        self.active = np.concatenate([layer.active for layer in layers],axis=2)
