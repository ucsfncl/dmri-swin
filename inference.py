import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage

from dipy.data import fetch_stanford_hardi, read_stanford_hardi, read_stanford_t1, read_stanford_pve_maps
from dipy.denoise.localpca import mppca
from dipy.denoise.patch2self import patch2self
from dipy.align.reslice import reslice
from dipy.io import image
from dipy.core import geometry

from monai.inferers import sliding_window_inference
from utilities import instantiate_from_config


def run_eval(X: torch.Tensor, model=None):
    mask = X[0, 0, :, :, :].bool()
    X = X[:, 1:, :, :, :]
    X[:, 0, mask] = (X[:, 0, mask] - torch.mean(X[:, 0, mask], dim=-1, keepdim=True)) / torch.std(X[:, 0, mask], dim=-1, keepdim=True)
    m = torch.mean(X[:, 1, mask], dim=-1, keepdim=True)
    s = torch.std(X[:, 1, mask], dim=-1, keepdim=True)

    X[:, 1, mask] = (X[:, 1, mask] - m) / s
    y = model(X)
    y[:, 0, ~mask] = 0.0
    y[:, 0, mask] = torch.clamp(y[:, 0, mask] * s + m, 0)
    return y

def swin_denoise(model, t1: np.ndarray, dwi: np.ndarray, bvals: np.ndarray, mask: np.ndarray, dwi_affine: np.ndarray = None, t1_affine: np.ndarray = None, resample: bool = True, resample_back: bool = True):
    if resample:
        dwi_scale, shear, angles, trans, persp = geometry.decompose_matrix(dwi_affine)
        mask_swin, swin_affine = reslice(mask, dwi_affine, np.abs(dwi_scale), (1.25, 1.25, 1.25), 0)
        dwi_swin, swin_affine = reslice(dwi, dwi_affine, np.abs(dwi_scale), (1.25, 1.25, 1.25), 5)
        mask_swin = (mask_swin > 0.5).astype(bool)
        t1_swin = ndimage.affine_transform(t1, np.linalg.inv(t1_affine) @ swin_affine, output_shape=mask_swin.shape, order=5)
        dwi_swin = dwi_swin[mask_swin, :]
        t1_swin  = t1_swin[mask_swin]
    else:
        mask_swin = mask
        dwi_swin = dwi[mask, :]
        t1_swin = t1[mask]
    t1_swin = np.clip(t1_swin, 0.0, None)
    dwi_swin[:, bvals < 100] = np.clip(dwi_swin[:, bvals < 100], 1.0, None)
    dwi_swin[:, bvals >= 100] = np.clip(dwi_swin[:, bvals >= 100], 0.0, None)

    X = np.zeros((dwi_swin.shape[-1], 3, mask_swin.shape[0], mask_swin.shape[1], mask_swin.shape[2]))
    for i in range(X.shape[0]):
        X[i, 0, :, :, :] = mask_swin
        X[i, 1, mask_swin] = t1_swin
        X[i, 2, mask_swin] = dwi_swin[:, i]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")
    model = model.to(device)
    model.eval()

    X = torch.from_numpy(X).to(device).float()
    with torch.no_grad():
        X_ret = sliding_window_inference(inputs=X, roi_size=[128, 128, 128], sw_batch_size=1, predictor=run_eval, overlap=0.875, progress=True, model=model)

    X_ret = X_ret[:, 0, :, :, :].detach().cpu().numpy()
    X_ret = np.transpose(X_ret, (1, 2, 3, 0))

    if resample_back:
        X_ret, _ = reslice(X_ret, swin_affine, (1.25, 1.25, 1.25), np.abs(dwi_scale), order=5)
        X_ret[~mask, :] = 0.0
        return X_ret, dwi_affine
    else:
        X_ret[~mask_swin, :] = 0.0
        return X_ret, swin_affine


def swin_denoise_low_mem(model, t1: np.ndarray, dwi: np.ndarray, bvals: np.ndarray, mask: np.ndarray, dwi_affine: np.ndarray = None, t1_affine: np.ndarray = None, resample: bool = True, resample_back: bool = True):
    if resample:
        dwi_scale, shear, angles, trans, persp = geometry.decompose_matrix(dwi_affine)
        mask_swin, swin_affine = reslice(mask, dwi_affine, np.abs(dwi_scale), (1.25, 1.25, 1.25), 0)
        dwi_swin, swin_affine = reslice(dwi, dwi_affine, np.abs(dwi_scale), (1.25, 1.25, 1.25), 5)
        mask_swin = (mask_swin > 0.5).astype(bool)
        t1_swin = ndimage.affine_transform(t1, np.linalg.inv(t1_affine) @ swin_affine, output_shape=mask_swin.shape, order=5)
        dwi_swin = dwi_swin[mask_swin, :]
        t1_swin  = t1_swin[mask_swin]
    else:
        mask_swin = mask
        dwi_swin = dwi[mask, :]
        t1_swin = t1[mask]
    t1_swin = np.clip(t1_swin, 0.0, None)
    dwi_swin[:, bvals < 100] = np.clip(dwi_swin[:, bvals < 100], 1.0, None)
    dwi_swin[:, bvals >= 100] = np.clip(dwi_swin[:, bvals >= 100], 0.0, None)

    X = np.zeros((dwi_swin.shape[-1], 3, mask_swin.shape[0], mask_swin.shape[1], mask_swin.shape[2]))
    for i in range(X.shape[0]):
        X[i, 0, :, :, :] = mask_swin
        X[i, 1, mask_swin] = t1_swin
        X[i, 2, mask_swin] = dwi_swin[:, i]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")
    model = model.to(device)
    model.eval()

    X_ret = np.zeros((dwi_swin.shape[-1], mask_swin.shape[0], mask_swin.shape[1], mask_swin.shape[2]))
    with torch.no_grad():
        for i in range(X.shape[0]):
            X_i = torch.from_numpy(X[None, i, :, :, :, :]).to(device).float()
            X_ret[i, :, :, :] = sliding_window_inference(inputs=X_i, roi_size=[128, 128, 128], sw_batch_size=1, predictor=run_eval, overlap=0.875, progress=True, model=model, device=torch.device('cpu'))[0, 0, :, :, :].numpy()
    X_ret = np.transpose(X_ret, (1, 2, 3, 0))

    if resample_back:
        X_ret, _ = reslice(X_ret, swin_affine, (1.25, 1.25, 1.25), np.abs(dwi_scale), order=5)
        X_ret[~mask, :] = 0.0
        return X_ret, dwi_affine
    else:
        X_ret[~mask_swin, :] = 0.0
        return X_ret, swin_affine


if __name__ == "__main__":
    fetch_stanford_hardi()
    csf_img, gm_img, wm_img = read_stanford_pve_maps()
    csf_pve = csf_img.get_fdata()
    gm_pve = gm_img.get_fdata()
    wm_pve = wm_img.get_fdata()
    mask = (wm_pve + gm_pve + csf_pve) > 0
    wm_mask = (wm_pve > 0.5) & mask
    gm_mask = (gm_pve > 0.5) & mask

    t1_img = read_stanford_t1()
    t1 = t1_img.get_fdata()
    t1_affine = t1_img.affine
    # t1, t1_affine = image.load_nifti("SUB1_t1.nii.gz")  # Better results if using the OG T1 with better resolution

    dwi_img, gtab = read_stanford_hardi()
    dwi_affine = dwi_img.affine
    bvecs = gtab.bvecs
    bvals = gtab.bvals
    dwi = dwi_img.get_fdata()

    # SWIN (tries to use GPU)
    config_filepath = "dmri-swin/models/swin_denoise.yaml"
    config = OmegaConf.load(config_filepath)

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(config.ckpt_path, map_location="cpu"))

    dwi_swin, swin_affine = swin_denoise(model, t1, dwi, bvals, mask, dwi_affine, t1_affine, resample=True, resample_back=True)
    image.save_nifti("stanford_swin.nii.gz", dwi_swin, swin_affine)