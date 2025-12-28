
import numpy as np
H, W = 256, 256
C = 16
arr = np.zeros((C, H, W), dtype='float32')
arr[0] = 300 + 2*np.random.randn(H,W)
arr[1] = 300 + 4*np.random.randn(H,W)
arr[2] = np.clip(0.2 + 0.05*np.random.randn(H,W), 0, 1)
arr[3] = np.clip(0.25 + 0.05*np.random.randn(H,W), 0, 1)
arr[4] = np.clip(0.3 + 0.05*np.random.randn(H,W), 0, 1)
arr[5] = np.clip(0.45 + 0.08*np.random.randn(H,W), 0, 1)
arr[6] = np.clip(0.2 + 0.05*np.random.randn(H,W), 0, 1)
arr[7] = np.clip(0.15 + 0.04*np.random.randn(H,W), 0, 1)
arr[8] = arr[6].copy(); arr[9] = arr[7].copy()
arr[10] = (arr[5]-arr[4])/(arr[5]+arr[4]+1e-6)
arr[11] = 0.97 + 0.02*((arr[10]-np.nanmin(arr[10]))/(np.nanmax(arr[10])-np.nanmin(arr[10])+1e-6))
arr[12] = (arr[6]-arr[5])/(arr[6]+arr[5]+1e-6)
arr[13] = np.abs(np.random.randn(H,W))*0.1
arr[14] = np.abs(np.random.randn(H,W))*0.1
arr[15] = (arr[1]-np.nanmin(arr[1]))/(np.nanmax(arr[1])-np.nanmin(arr[1])+1e-6)
np.savez_compressed('sample_stack.npz', arr=arr, transform=(444720.0,30.0,0.0,3751320.0,0.0,-30.0), crs=32643, meta={'sensor':'simulated','date':'2025-12-01'})
print('Wrote sample_stack.npz')
