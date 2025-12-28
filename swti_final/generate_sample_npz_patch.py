
import numpy as np
def make_sample(path='sample_patch.npz'):
    H,W = 256,256; C=16
    arr = np.zeros((C,H,W), dtype='float32')
    base = 290 + np.linspace(0,4,W)[None,:] + 0.5*np.random.randn(H,W)
    arr[1] = base + 0.1*np.random.randn(H,W)
    arr[10] = (np.clip(0.2 + 0.3*np.random.rand(H,W), -1, 1))
    arr[11] = 0.97 + 0.02*((arr[10]-np.nanmin(arr[10]))/(np.nanmax(arr[10])-np.nanmin(arr[10])+1e-9))
    meta = {'date':'2025-12-08T10:00:00'}
    np.savez_compressed(path, patch=arr, meta=meta)
    print('Wrote', path)
if __name__ == '__main__':
    make_sample('sample_patch.npz')
