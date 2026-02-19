import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MipmappedTextureHighPerf(nn.Module):
    def __init__(self, base_height=512, base_width=512, num_levels=8, channels=16):
        super().__init__()
        self.num_levels = num_levels

        # Construct a multi - layer mipmap
        mipmaps = []
        h, w = base_height, base_width
        for i in range(num_levels):
            tex = nn.Parameter(torch.randn(1, channels, h, w) * 0.01)
            mipmaps.append(tex)
            h = max(h // 2, 1)
            w = max(w // 2, 1)

        self.mipmaps = nn.ParameterList(mipmaps)

    def forward(self, uv, p):
        """
        uv: (N, 2), Value [0,1]
        p:  (N,) or scalar, Value [0,1]
        return: (N, C)
        """
        N = uv.shape[0]

        # ensure p The shape and shape uv Alignment
        if isinstance(p, float) or isinstance(p, int):
            p = torch.tensor([p], dtype=uv.dtype, device=uv.device)
        if p.dim() == 0:  # scalar tensor
            p = p.unsqueeze(0).expand(N)
        elif p.dim() == 1 and p.size(0) == 1:
            p = p.expand(N)

        # 1) Constructing a one -time batch grid: (N, 1, 1, 2)
        #    [0,1] -> [-1,1]
        grid_2d = 2.0 * uv.unsqueeze(0).unsqueeze(2) - 1.0  # (1,N,1,2)

        # 2) Separate 9 Layer disposable batch grid_sample => get (9, N, C)
        #    - Every layer shape: (C, H_i, W_i) -> (1, C, H_i, W_i)
        #    - grid_sample -> (N, C, 1, 1)
        #    - Re -become (N, C) Deposit big_array[i]
        big_array = []
        for i in range(self.num_levels):
            # (C, H_i, W_i) -> (1, C, H_i, W_i)
            tex_2d = self.mipmaps[i]
            # Batch interpolation
            # Note: Here the BATCH dimension of GRID_2D is n, and the BATCH dimension of TEX_2D is 1
            # Pytorch allows automatic broadcast, so it will actual extend the Tex_2D to (n, c, h_i, W_i)
            sampled = F.grid_sample(
                tex_2d, grid_2d,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # (1, C, N, 1)

            sampled = sampled.squeeze() # (N, C)
            big_array.append(sampled.permute(1,0))

        # Stack => (num_levels, N, C)
        big_array = torch.stack(big_array, dim=0)  # (9, N, C)

        # 3)Calculate Level0, Level1 and interpolation factor Alpha corresponding to each sample
        level_float = p * (self.num_levels - 1)  # in [0, 8]
        level0 = torch.floor(level_float).long()                 # (N,)
        level1 = torch.clamp(level0 + 1, max=self.num_levels-1)  # (N,)
        alpha = level_float - level0.float()                     # (N,)

        # 4) From the results corresponding to each sample of Gather in Big_array
        # big_array SHAPE is (9, n, c)
        # We want to be right dim=0 (levels) Do indexes, but the level0 of each sample is different
        # Can take advantage of advanced indexing:
        #  - big_array[level0[i], i] The result of the first sample on the level0 [i]
        #  - big_array[level1[i], i] The result of the first sample in level1 [i]
        # In the end, all merged (N, C)
        #
        # There are many ways to write. The following way is given:
        idx0 = level0.unsqueeze(-1).expand(-1, big_array.size(-1))  # (N, C) But C is just a radio
        idx1 = level1.unsqueeze(-1).expand(-1, big_array.size(-1))

        # Use Gather "Manual" method:
        # OUT0 [i,:] = big_array [level0 [i], i,:]
        # OUT1 [i,:] = big_array [level1 [i], i,:]
        # (N, 9, c), then gather dim = 1
        big_array_T = big_array.permute(1, 0, 2)  # (N, 9, C)
        out0 = torch.gather(big_array_T, 1, idx0.unsqueeze(1)).squeeze(1)  # (N, C)
        out1 = torch.gather(big_array_T, 1, idx1.unsqueeze(1)).squeeze(1)  # (N, C)

        # 5) Do linear interpolation between OUT0 and OUT1
        alpha = alpha.unsqueeze(-1)  # (N,1)
        output = out0 * (1 - alpha) + out1 * alpha  # (N, C)

        return output


class MipmappedTexture3DHighPerf(nn.Module):
    # def __init__(self, base_height=512, base_width=1024, num_levels=9, channels=16):  # paper values
    def __init__(self, base_height=512, base_width=512, num_levels=8, channels=16):  # empirical values
        super().__init__()
        self.num_levels = num_levels
        self.channels = channels
        self.base_height = base_height
        self.base_width = base_width

        # (C, H, W) for each level
        mipmaps = []
        h, w = base_height, base_width

        self.tex = nn.Parameter(torch.randn(1, channels, h, w) * 0.01)

    def visualization(self):
        # Convert 16-channel feature to 3-channel RGB using learned weights
        # Use first 3 channels directly as RGB
        rgb = self.tex[:, :3, :, :]  # (1, 3, H, W)

        # Normalize to [0,1] range for visualization
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = rgb[0]

        return rgb

    def forward(self, uv, p):
        """
        uv: (N, 2), value [0,1]
        p:  (N,) or scalar, value [0,1]
        return: (N, C)
        """
        N = uv.shape[0]

        # Ensure that the shape of P is aligned with UV
        if isinstance(p, float) or isinstance(p, int):
            p = torch.tensor([p], dtype=uv.dtype, device=uv.device)
        if p.dim() == 0:  # scalar tensor
            p = p.unsqueeze(0).expand(N)
        elif p.dim() == 1 and p.size(0) == 1:
            p = p.expand(N)

        # 1) Constructing a one -time batch grid: (N, 1, 1, 2)
        #    [0,1] -> [-1,1]
        grid_2d = 2.0 * uv.unsqueeze(0).unsqueeze(2) - 1.0  # (1,N,1,2)

        # 2) Obtain (9, n, C)
        #    - Every layer shape: (C, H_i, W_i) -> (1, C, H_i, W_i)
        #    - grid_sample -> (N, C, 1, 1)
        #    - Re -become (N, C) Deposit big_array[i]
        big_array = []
        for i in range(self.num_levels):
            # (C, H_i, W_i) -> (1, C, H_i, W_i)
            h = max(self.base_height // 2**(i), 1)
            w = max(self.base_width // 2**(i), 1)
            tex_2d = F.interpolate(
                self.tex,
                size=(h, w),
                mode='bilinear'
            )  # (1, C, H, W)
            # Batch interpolation
            # notice: here grid_2d of batch Dimension N, and tex_2d of batch Dimension 1
            # PyTorch Allow automatic Broadcast, so it will actually expand tex_2d arrive (N, C, H_i, W_i)
            sampled = F.grid_sample(
                tex_2d, grid_2d,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # (1, C, N, 1)

            sampled = sampled.squeeze() # (N, C)
            big_array.append(sampled.permute(1,0))

        # Stack => (num_levels, N, C)
        big_array = torch.stack(big_array, dim=0)  # (9, N, C)

        # 3) Calculate the corresponding of each sample level0、level1 Interpolation factor alpha
        level_float = p * (self.num_levels - 1)  # in [0, 8]
        level0 = torch.floor(level_float).long()                 # (N,)
        level1 = torch.clamp(level0 + 1, max=self.num_levels-1)  # (N,)
        alpha = level_float - level0.float()                     # (N,)

        # 4)From the results corresponding to each sample of Gather in Big_array
        # Big_array's shape is (9, n, c)
        # We want to index on dim = 0 (levels), but the level0 of each sample is different
        # You can use Advanced Indexing:
        #  - big_array [level0 [i], i] gets the result of the i -i sample in level0 [i]
        #  - big_array [level1 [i], i] gets the result of the first sample in Level1 [i]
        # In the end, all merged (N, C)
        #
        # There are many ways to write. The following way is given:
        idx0 = level0.unsqueeze(-1).expand(-1, big_array.size(-1))  # (N, C) 但C只是做广播
        idx1 = level1.unsqueeze(-1).expand(-1, big_array.size(-1))

        # Use the Gather "manual" method:
        #   out0[i, :] = big_array[level0[i], i, :]
        #   out1[i, :] = big_array[level1[i], i, :]
        # First turn to (n, 9, c), then gather dim = 1
        big_array_T = big_array.permute(1, 0, 2)  # (N, 9, C)
        out0 = torch.gather(big_array_T, 1, idx0.unsqueeze(1)).squeeze(1)  # (N, C)
        out1 = torch.gather(big_array_T, 1, idx1.unsqueeze(1)).squeeze(1)  # (N, C)

        # 5) Do linear interpolation between OUT0 and OUT1
        alpha = alpha.unsqueeze(-1)  # (N,1)
        output = out0 * (1 - alpha) + out1 * alpha  # (N, C)

        return output


# ---------------------- test ----------------------------------
if __name__ == "__main__":
    model_3d = MipmappedTexture3DHighPerf()
    model_3d.cuda().eval()

    N = 8192
    uv_test = torch.rand(N, 2, device='cuda')
    p_test = torch.rand(N, device='cuda')

    for _ in range(5):
        out_3d = model_3d(uv_test, p_test)
    print("3D Output shape:", out_3d.shape)

'''
# ---------------------- Test and compare performance ----------------------------------
if __name__ == "__main__":
    model = MipmappedTextureHighPerf(
        base_height=256,
        base_width=256,
        num_levels=9,
        channels=16
    )

    # Be a big batch
    N = 8192
    uv_test = torch.rand(N, 2, device='cuda')
    p_test = torch.rand(N, device='cuda')

    model.cuda()
    model.eval()

    # Run a few times, look at the speed
    for _ in range(5):
        out = model(uv_test, p_test)  # (N, 16)
    print("Output shape:", out.shape)
'''