import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from dipy.data import fetch_stanford_hardi, read_stanford_hardi, read_stanford_t1, read_stanford_pve_maps
from dipy.denoise.localpca import mppca
from dipy.denoise.patch2self import patch2self
from dipy.reconst import dti
from dipy.core.gradients import GradientTable

from utilities import instantiate_from_config
from inference import swin_denoise


def dti_metrics(diff_tensor: np.ndarray):
    evalues, evectors = np.linalg.eigh(diff_tensor)

    fa = np.sqrt(0.5 * (np.square(evalues[:, 0] - evalues[:, 1]) + np.square(evalues[:, 0] - evalues[:, 2]) + np.square(evalues[:, 1] - evalues[:, 2])) / np.sum(np.square(evalues), axis=-1))
    md = np.mean(evalues, axis=-1)
    rd = (evalues[:, 0] + evalues[:, 1]) / 2
    ad = evalues[:, 2]
    v1 = evectors[:, :, 2]
    return {"fa": fa, "md": md, "rd": rd, "ad": ad, "v1": v1}


def fit_dti(b0, shell, bvals, bvecs, ret_quadratic=True):
    shell = np.clip(np.clip(shell, 0, None) / np.clip(b0[:, None], 1.0, None), 1e-5, 1.0)
    dwi = np.concatenate((np.ones(shell.shape[0])[:, None], shell), axis=-1)

    bvals = np.concatenate((np.zeros(1), bvals))
    bvecs = np.concatenate((np.zeros((1, 3)), bvecs))

    gtab = GradientTable(bvals[:, None] * bvecs)
    dti_model = dti.TensorModel(gtab)
    dti_fit = dti_model.fit(dwi)
    if ret_quadratic:
        return dti_fit.quadratic_form
    else:
        return dti_fit
    

def make_dti_results(gt_dti, ret_dti, mask):
    ret_dict = dti_metrics(ret_dti[mask, :, :])
    gt_dict = dti_metrics(gt_dti[mask, :, :])

    mae_dict = {
        "fa": np.mean(np.abs(ret_dict["fa"] - gt_dict["fa"])),
        "md": np.mean(np.abs(ret_dict["md"] - gt_dict["md"])),
        "rd": np.mean(np.abs(ret_dict["rd"] - gt_dict["rd"])),
        "ad": np.mean(np.abs(ret_dict["ad"] - gt_dict["ad"])),
        "v1": np.mean(np.abs(np.arccos(np.abs(np.sum(ret_dict["v1"] * gt_dict["v1"], axis=-1))))) * 180 / np.pi
        }

    return mae_dict


def rotate(g, theta_x, theta_y, theta_z):
    R_x = np.zeros((3, 3))
    R_y = np.zeros((3, 3))
    R_z = np.zeros((3, 3))

    a_x = np.cos(theta_x)
    b_x = np.sin(theta_x)
    a_y = np.cos(theta_y)
    b_y = np.sin(theta_y)
    a_z = np.cos(theta_z)
    b_z = np.sin(theta_z)

    R_x[0, 0] = 1.0
    R_x[1, 1] = a_x
    R_x[2, 1] = b_x
    R_x[1, 2] = -b_y
    R_x[2, 2] = a_x

    R_y[1, 1] = 1.0
    R_y[0, 0] = a_y
    R_y[2, 0] = -b_y
    R_y[0, 2] = b_y
    R_y[2, 2] = a_y

    R_z[2, 2] = 1.0
    R_z[0, 0] = a_z
    R_z[1, 0] = b_z
    R_z[0, 1] = -b_z
    R_z[1, 1] = a_z

    R = np.dot(np.dot(R_x, R_y), R_z)
    g_rot = np.dot(g, R.transpose())
    return g_rot


def find_best_indices(g, g_opt):
    cos_sim = np.abs(np.dot(g, g_opt.transpose()))
    best_indices = np.argmax(cos_sim, axis=0)
    return best_indices


def find_cond(x):
    mat = np.concatenate((np.square(x[:, 0]).reshape(-1, 1), (2 * x[:, 0] * x[:, 1]).reshape(-1, 1), (2 * x[:, 0] * x[:, 2]).reshape(-1, 1), np.square(x[:, 1]).reshape(-1, 1), (2 * x[:, 1] * x[:, 2]).reshape(-1, 1), np.square(x[:, 2]).reshape(-1, 1)), axis=1)
    cond = np.linalg.cond(mat)
    return cond


def find_best_subset(G, n: int = 20):
    a = 0.910
    b = 0.416
    g_opt = np.array([[a, b, 0], [0, a, b], [b, 0, a], [a, -b, 0], [0, a, -b], [-b, 0, a]])

    best_indices = None
    best_cond = 1e12

    theta_x = np.linspace(0, np.pi, n, endpoint=False)
    theta_y = np.linspace(0, np.pi, n , endpoint=False)
    theta_z = np.linspace(0, np.pi, n, endpoint=False)

    a = time.perf_counter()

    for i in range(theta_x.shape[0]):
        for j in range(theta_y.shape[0]):
            for k in range(theta_z.shape[0]):
                current_opt = rotate(g_opt, theta_x[i], theta_y[j], theta_z[k])
                current_indices = find_best_indices(G, current_opt)
                current_cond = find_cond(G[current_indices, :])
                if current_cond < best_cond:
                    best_cond = current_cond
                    best_indices = current_indices
    
    b = time.perf_counter()
    print(f"Took {b - a} seconds to find the best 6-direction subset")
    print(f"Condition is {best_cond}")
    print(best_indices)
    return best_indices


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
    # t1, t1_affine = image.load_nifti("SUB1_t1.nii.gz")

    dwi_img, gtab = read_stanford_hardi()
    dwi_affine = dwi_img.affine
    bvecs = gtab.bvecs
    bvals = gtab.bvals
    dwi = dwi_img.get_fdata()

    ret = {}

    # GT
    b0 = dwi[:, :, :, bvals == 0]
    b2000 = dwi[:, :, :, bvals == 2000]
    b2000_bvecs = bvecs[bvals == 2000, :]
    ret["gt"] = fit_dti(np.mean(b0[mask, :], axis=-1), b2000[mask, :], np.ones(b2000_bvecs.shape[0]) * 2e3, b2000_bvecs)


    # SUBSET
    # best_indices = find_best_subset(b2000_bvecs, n=200)  # Should be [134  69  22  15   8  35] in no particular order
    best_indices = [134, 69, 22, 15, 8, 35]
    bvals_subset = np.ones(6) * 2e3
    bvecs_subset = b2000_bvecs[best_indices, :]

    b0_subset = b0[:, :, :, 0]
    b2000_subset = b2000[:, :, :, best_indices]
    dwi_subset = np.concatenate((b0_subset[:, :, :, None], b2000_subset), axis=-1)


    # RAW (No Denoising)
    ret["raw"] = fit_dti(b0_subset[mask], b2000_subset[mask, :], bvals_subset, bvecs_subset)
    print("MAE for RAW (no denoising)")
    print(make_dti_results(ret["gt"], ret["raw"], wm_mask[mask]))
    print(make_dti_results(ret["gt"], ret["raw"], gm_mask[mask]))


    # MPPCA
    dwi_mppca = mppca(dwi_subset, mask, patch_radius=3)
    ret["mppca"] = fit_dti(dwi_mppca[mask, 0], dwi_mppca[mask, 1:], bvals_subset, bvecs_subset)
    print("MAE for MPPCA denoising")
    print(make_dti_results(ret["gt"], ret["mppca"], wm_mask[mask]))
    print(make_dti_results(ret["gt"], ret["mppca"], gm_mask[mask]))

    # P2S
    dwi_p2s = patch2self(dwi_subset, np.concatenate((np.zeros(1), bvals_subset)))
    ret["p2s"] = fit_dti(dwi_p2s[mask, 0], dwi_p2s[mask, 1:], bvals_subset, bvecs_subset)
    print("MAE for P2S denoising")
    print(make_dti_results(ret["gt"], ret["p2s"], wm_mask[mask]))
    print(make_dti_results(ret["gt"], ret["p2s"], gm_mask[mask]))

    # SWIN (tries to use GPU)
    config_filepath = "dmri-swin/swin_denoise.yaml"
    config = OmegaConf.load(config_filepath)

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(config.ckpt_path, map_location="cpu"))

    dwi_swin, swin_affine = swin_denoise(model, t1, dwi_subset, np.array([0.0, 2e3, 2e3, 2e3, 2e3, 2e3, 2e3]), mask, dwi_affine, t1_affine, resample=True, resample_back=True)
    ret["swin"] = fit_dti(dwi_swin[mask, 0], dwi_swin[mask, 1:], bvals_subset, bvecs_subset)

    print("MAE for SWIN denoising")
    print(make_dti_results(ret["gt"], ret["swin"], wm_mask[mask]))
    print(make_dti_results(ret["gt"], ret["swin"], gm_mask[mask]))


    # Plot DTI Metrics
    metrics = ["fa", "ad", "rd", "md"]
    methods = ["gt", "raw", "p2s", "mppca", "swin"]

    img_coord = np.concatenate([s[:, None] for s in np.where(mask)], axis=1)
    coord_max = np.max(img_coord, axis=0) + 1
    coord_min = np.min(img_coord, axis=0)
    mask = mask[coord_min[0]: coord_max[0], coord_min[1]: coord_max[1], coord_min[2]: coord_max[2]]
    
    k_index = mask.shape[2] // 2
    x_size = len(methods) * mask.shape[0]
    y_size = len(metrics) * mask.shape[1]
    r = 10.0 / y_size
    x_size = x_size * r
    y_size = y_size * r
    fig, axes = plt.subplots(nrows=len(metrics), ncols=len(methods), sharex="all", sharey="all", gridspec_kw={"wspace": 0.0, "hspace": 0.0}, figsize = (x_size, y_size))

    for method_idx, method in enumerate(methods):
        ret_metrics = dti_metrics(ret[method])
        color_fa = np.zeros(mask.shape + (3,))
        ad = np.zeros(mask.shape)
        rd = np.zeros(mask.shape)
        md = np.zeros(mask.shape)

        color_fa[mask, :] = np.clip(ret_metrics["fa"][:, None], 0.0, 1.0) * np.abs(ret_metrics["v1"])
        ad[mask] = np.clip(ret_metrics["ad"], 0.0, 3e-3)
        rd[mask] = np.clip(ret_metrics["rd"], 0.0, 3e-3)
        md[mask] = np.clip(ret_metrics["md"], 0.0, 3e-3)

        color_fa = np.rot90(color_fa[:, :, k_index, :])
        ad = np.rot90(ad[:, :, k_index])
        rd = np.rot90(rd[:, :, k_index])
        md = np.rot90(md[:, :, k_index])
        
        axes[0, method_idx].imshow(color_fa, vmin=0, vmax=1)
        axes[1, method_idx].imshow(ad, cmap="gray", vmin=0, vmax=3e-3)
        axes[2, method_idx].imshow(rd, cmap="gray", vmin=0, vmax=3e-3)
        axes[3, method_idx].imshow(md, cmap="gray", vmin=0, vmax=3e-3)

    for idx, metric in enumerate(metrics):
        axes[idx, 0].set_ylabel(metric.upper())

    for idx, method in enumerate(methods):
        axes[3, idx].set_xlabel(method.upper())

    for i in range(len(metrics)):
        for j in range(len(methods)):
            axes[i, j].set_xticks([])
            axes[i, j].set_xticks([], minor=True)
            axes[i, j].set_yticks([])
            axes[i, j].set_yticks([], minor=True)
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['bottom'].set_visible(False)
            axes[i, j].spines['left'].set_visible(False)
    axes[0, 2].set_title("Stanford 6 Direction DTI Metrics", fontsize=16)
    plt.tight_layout()
    fig.savefig("dmri-swin/figs/stanford_dti_metrics.png", bbox_inches='tight', dpi=300)