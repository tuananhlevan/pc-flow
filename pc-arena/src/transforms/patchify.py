import random


class Patchify():
    def __init__(self, patch_size, aligned = False, channel_last = False):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) or isinstance(patch_size, list)
        self.aligned = aligned
        self.channel_last = channel_last

    def __call__(self, x):
        if self.channel_last:
            H, W = x.size(0), x.size(1)
        else:
            H, W = x.size(1), x.size(2)

        if not self.aligned:
            Hs, Ws = random.randint(0, H - self.patch_size[0]), random.randint(0, W - self.patch_size[1])
            He, We = Hs + self.patch_size[0], Ws + self.patch_size[1]
        else:
            Hs, Ws = random.randint(0, H // self.patch_size[0] - 1) * self.patch_size[0], random.randint(0, W // self.patch_size[1] - 1) * self.patch_size[1]
            He, We = Hs + self.patch_size[0], Ws + self.patch_size[1]

        if self.channel_last:
            return x[Hs:He,Ws:We,:]
        else:
            return x[:,Hs:He,Ws:We]