import mmcv
import torch
import numpy as np

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['masked_im2col_forward', 'masked_col2im_forward'])

def generator():
  H = [240, 20, 20]
  W = [240, 20, 20]
  C = [1024, 256, 256]
  mask_cnt = [1000, 10000, 200]
  kernel_h_ = [8, 3, 3]
  kernel_w_ = [8, 3, 3]
  pad_h_ = [1, 1, 1]
  pad_w_ = [1, 1, 1]
  for i in range(len(H)):
    x = torch.randn((1,C[i],H[i],W[i]), dtype=torch.float).cuda()

    torch_mask_h_idx = torch.randint(0, H[i], (mask_cnt[i],), dtype = torch.int64).cuda()
    torch_mask_w_idx = torch.randint(0, W[i], (mask_cnt[i],), dtype = torch.int64).cuda()
    y = torch.zeros((C[i] * kernel_h_[i] * kernel_w_[i], mask_cnt[i]), dtype=torch.float).cuda()
    ext_module.masked_im2col_forward(
      x,
      torch_mask_h_idx,
      torch_mask_w_idx,
      y,
      kernel_h = kernel_h_[i],
      kernel_w = kernel_w_[i],
      pad_h = pad_h_[i],
      pad_w = pad_w_[i])

    # print(y)
    np.savez("masked_im2col2"+str(i)+".npz", x = x.cpu().detach().numpy(),
      torch_mask_h_idx = torch_mask_h_idx.cpu().detach().numpy(),
      torch_mask_w_idx = torch_mask_w_idx.cpu().detach().numpy(),
      y = y.cpu().detach().numpy())

def load():

  H = [240, 20, 20]
  W = [240, 20, 20]
  C = [1024, 256, 256]
  mask_cnt = [1000, 10000, 200]
  kernel_h_ = [8, 3, 3]
  kernel_w_ = [8, 3, 3]
  pad_h_ = [1, 1, 1]
  pad_w_ = [1, 1, 1]
  for i in range(len(H)):
    datas = np.load("masked_im2col2"+str(i)+".npz")
    np_x = datas['x']
    np_mask_h_idx = datas['torch_mask_h_idx']
    np_mask_w_idx = datas['torch_mask_w_idx']
    np_y = datas['y']
    x = torch.tensor(np_x, dtype=torch.float).cuda()
    torch_mask_h_idx = torch.tensor(np_mask_h_idx, dtype = torch.int64).cuda()
    torch_mask_w_idx = torch.tensor(np_mask_w_idx, dtype = torch.int64).cuda()
    torch_y = torch.tensor(np_y, dtype=torch.float).cuda()
    y = torch.zeros((C[i] * kernel_h_[i] * kernel_w_[i], mask_cnt[i]), dtype=torch.float).cuda()
    for _ in range(5):
      ext_module.masked_im2col_forward(
        x,
        torch_mask_h_idx,
        torch_mask_w_idx,
        y,
        kernel_h = kernel_h_[i],
        kernel_w = kernel_w_[i],
        pad_h = pad_h_[i],
        pad_w = pad_w_[i])
      torch.cuda.synchronize()
    assert torch.all(y == torch_y)
    print('*********************')
if __name__ == "__main__" :
  print('*****generator*****')
  generator()
  torch.cuda.synchronize()
  print('*******load********')
  load()