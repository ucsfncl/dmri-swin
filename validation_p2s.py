import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
import time
from omegaconf import OmegaConf

from dipy.data import fetch_stanford_hardi, read_stanford_hardi, read_stanford_t1, read_stanford_pve_maps
from dipy.denoise.localpca import mppca
from dipy.denoise.patch2self import patch2self
from dipy.io import image
from dipy.reconst import dti
from dipy.core.gradients import GradientTable
import dipy.reconst.cross_validation as xval
import dipy.reconst.csdeconv as csd

from utilities import instantiate_from_config
from inference import swin_denoise



def dti_metrics(diff_tensor: np.ndarray):
    evalues, evectors = np.linalg.eigh(diff_tensor)

    fa = np.sqrt(0.5 * (np.square(evalues[:, 0] - evalues[:, 1]) + np.square(evalues[:, 0] - evalues[:, 2]) + np.square(evalues[:, 1] - evalues[:, 2])) / np.sum(np.square(evalues), axis=-1))
    md = np.mean(evalues, axis=-1)
    rd = (evalues[:, 0] + evalues[:, 1]) / 2
    ad = evalues[:, 2]
    v1 = evectors[:, :, 2]
    return fa, md, rd, ad, v1

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
    ret_fa, ret_md, ret_rd, ret_ad, ret_v1 = dti_metrics(ret_dti[mask, :, :])
    gt_fa, gt_md, gt_rd, gt_ad, gt_v1 = dti_metrics(gt_dti[mask, :, :])

    mae_dict = {
        "fa": np.mean(np.abs(ret_fa - gt_fa)),
        "md": np.mean(np.abs(ret_md - gt_md)),
        "rd": np.mean(np.abs(ret_rd - gt_rd)),
        "ad": np.mean(np.abs(ret_ad - gt_ad)),
        "v1": np.mean(np.abs(np.arccos(np.abs(np.sum(ret_v1 * gt_v1, axis=-1))))) * 180 / np.pi
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

    # RAW
    image.save_nifti("stanford_raw.nii.gz", dwi, dwi_affine)

    # MPPCA
    dwi_mppca = mppca(dwi, mask, patch_radius=3)
    image.save_nifti("stanford_mppca.nii.gz", dwi_mppca, dwi_affine)

    # P2S
    dwi_p2s = patch2self(dwi, bvals)
    image.save_nifti("stanford_p2s.nii.gz", dwi_p2s, dwi_affine)

    # SWIN
    config_filepath = "dmri-swin/swin_denoise.yaml"
    config = OmegaConf.load(config_filepath)

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(config.ckpt_path, map_location="cpu"))
    dwi_swin, swin_affine = swin_denoise(model, t1, dwi, bvals, mask, dwi_affine, t1_affine, resample=True, resample_back=True)
    image.save_nifti("stanford_swin.nii.gz", dwi_swin, swin_affine)

    np.random.seed(0)  # pick random seed to have reproducible results
    stanford_raw, affine = image.load_nifti("stanford_raw.nii.gz")
    stanford_mppca, affine = image.load_nifti("stanford_mppca.nii.gz")
    stanford_p2s, affine = image.load_nifti("stanford_p2s.nii.gz")
    stanford_swin, affine = image.load_nifti("stanford_swin.nii.gz")

    cc_vox = stanford_raw[40, 70, 38]
    cso_vox = stanford_raw[30, 76, 38]

    cc_vox_mp = stanford_mppca[40, 70, 38]
    cso_vox_mp = stanford_mppca[30, 76, 38]

    cc_vox_p2s = stanford_p2s[40, 70, 38]
    cso_vox_p2s = stanford_p2s[30, 76, 38]

    cc_vox_swin = stanford_swin[40, 70, 38]
    cso_vox_swin = stanford_swin[30, 76, 38]

    dti_model = dti.TensorModel(gtab)
    response, ratio = csd.auto_response_ssst(gtab, stanford_raw, roi_radii=10, fa_thr=0.7)
    csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response)

    dti_model = dti.TensorModel(gtab)
    response_p2s, ratio = csd.auto_response_ssst(gtab, stanford_p2s, roi_radii=10, fa_thr=0.7)
    csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response)

    dti_model = dti.TensorModel(gtab)
    response_mp, ratio = csd.auto_response_ssst(gtab, stanford_mppca, roi_radii=10, fa_thr=0.7)
    csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response)

    dti_model = dti.TensorModel(gtab)
    response_swin, ratio = csd.auto_response_ssst(gtab, stanford_swin, roi_radii=10, fa_thr=0.7)
    csd_model = csd.ConstrainedSphericalDeconvModel(gtab, response)


    dti_cc = xval.kfold_xval(dti_model, cc_vox, 3)
    csd_cc = xval.kfold_xval(csd_model, cc_vox, 3, response)
    dti_cso = xval.kfold_xval(dti_model, cso_vox, 3)
    csd_cso = xval.kfold_xval(csd_model, cso_vox, 3, response)

    dti_cc_P2S = xval.kfold_xval(dti_model, cc_vox_p2s, 3)
    csd_cc_P2S = xval.kfold_xval(csd_model, cc_vox_p2s, 3, response_p2s)
    dti_cso_P2S = xval.kfold_xval(dti_model, cso_vox_p2s, 3)
    csd_cso_P2S = xval.kfold_xval(csd_model, cso_vox_p2s, 3, response_p2s)

    dti_cc_MP = xval.kfold_xval(dti_model, cc_vox_mp, 3)
    csd_cc_MP = xval.kfold_xval(csd_model, cc_vox_mp, 3, response_mp)
    dti_cso_MP = xval.kfold_xval(dti_model, cso_vox_mp, 3)
    csd_cso_MP = xval.kfold_xval(csd_model, cso_vox_mp, 3, response_mp)


    dti_cc_SWIN = xval.kfold_xval(dti_model, cc_vox_swin, 3)
    csd_cc_SWIN = xval.kfold_xval(csd_model, cc_vox_swin, 3, response_swin)
    dti_cso_SWIN = xval.kfold_xval(dti_model, cso_vox_swin, 3)
    csd_cso_SWIN = xval.kfold_xval(csd_model, cso_vox_swin, 3, response_swin)

    cc_dti_r2 = stats.pearsonr(cc_vox[~gtab.b0s_mask], dti_cc[~gtab.b0s_mask])[0]**2
    cc_dti_r2_mp = stats.pearsonr(cc_vox[~gtab.b0s_mask], dti_cc_MP[~gtab.b0s_mask])[0]**2
    cc_dti_r2_p2s = stats.pearsonr(cc_vox[~gtab.b0s_mask], dti_cc_P2S[~gtab.b0s_mask])[0]**2
    cc_dti_r2_swin = stats.pearsonr(cc_vox[~gtab.b0s_mask], dti_cc_SWIN[~gtab.b0s_mask])[0]**2

    cso_dti_r2 = stats.pearsonr(cso_vox[~gtab.b0s_mask], dti_cso[~gtab.b0s_mask])[0]**2
    cso_dti_r2_mp = stats.pearsonr(cso_vox[~gtab.b0s_mask], dti_cso_MP[~gtab.b0s_mask])[0]**2
    cso_dti_r2_p2s = stats.pearsonr(cso_vox[~gtab.b0s_mask], dti_cso_P2S[~gtab.b0s_mask])[0]**2
    cso_dti_r2_swin = stats.pearsonr(cso_vox[~gtab.b0s_mask], dti_cso_SWIN[~gtab.b0s_mask])[0]**2
    
    print(f"Corpus Callosum\n DTI Raw R2: {cc_dti_r2}\n DTI MPPCA R2: {cc_dti_r2_mp}\n DTI P2S R2: {cc_dti_r2_p2s}\n DTI SWIN R2: {cc_dti_r2_swin}\n")
    print(f"Centrum Semiovale\n DTI Raw R2: {cso_dti_r2}\n DTI MPPCA R2: {cso_dti_r2_mp}\n DTI P2S R2: {cso_dti_r2_p2s}\n DTI SWIN R2: {cso_dti_r2_swin}\n")


    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches([15, 7])

    # the R2 values in plots may vary due to random seed
    ax[0].plot(cc_vox[~gtab.b0s_mask], dti_cc[~gtab.b0s_mask], 'X', color='crimson', label='Original: $R^{2}=' + f'{cc_dti_r2:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[0].plot(cc_vox_mp[~gtab.b0s_mask], dti_cc_MP[~gtab.b0s_mask], 'o', color='darkorange', label='Marchencko Pastur: $R^{2}=' + f'{cc_dti_r2_mp:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[0].plot(cc_vox_p2s[~gtab.b0s_mask], dti_cc_P2S[~gtab.b0s_mask], 'D', color='teal', label='Patch2Self: $R^{2}=' + f'{cc_dti_r2_p2s:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[0].plot(cc_vox_swin[~gtab.b0s_mask], dti_cc_SWIN[~gtab.b0s_mask], 's', color='darkviolet', label='SWIN: $R^{2}=' + f'{cc_dti_r2_swin:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')

    ax[1].plot(cso_vox[~gtab.b0s_mask], dti_cso[~gtab.b0s_mask], 'X', color='crimson', label='Original: $R^{2}=' + f'{cso_dti_r2:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[1].plot(cso_vox_mp[~gtab.b0s_mask], dti_cso_MP[~gtab.b0s_mask], 'o', color='darkorange', label='Marchencko Pastur: $R^{2}=' + f'{cso_dti_r2_mp:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[1].plot(cso_vox_p2s[~gtab.b0s_mask], dti_cso_P2S[~gtab.b0s_mask], 'D', color='teal', label='Patch2Self: $R^{2}=' + f'{cso_dti_r2_p2s:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[1].plot(cso_vox_swin[~gtab.b0s_mask], dti_cso_SWIN[~gtab.b0s_mask], 's', color='darkviolet', label='SWIN: $R^{2}=' + f'{cso_dti_r2_swin:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')

    ax[0].set_title('Corpus Callosum')
    ax[1].set_title('Centrum Semiovale')
    for this_ax in ax:
        this_ax.set_xlabel('Data')
        this_ax.set_ylabel('Model prediction')
        this_ax.legend(loc='upper left', fontsize='large')
    fig.suptitle("DTI Cross Validation")
    fig.savefig("dmri-swin/figs/dti_cross_val.png", dpi=300, bbox_inches='tight')

    cc_csd_r2 = stats.pearsonr(cc_vox[~gtab.b0s_mask], csd_cc[~gtab.b0s_mask])[0]**2
    cc_csd_r2_mp = stats.pearsonr(cc_vox_mp[~gtab.b0s_mask], csd_cc_MP[~gtab.b0s_mask])[0]**2
    cc_csd_r2_p2s = stats.pearsonr(cc_vox_p2s[~gtab.b0s_mask], csd_cc_P2S[~gtab.b0s_mask])[0]**2
    cc_csd_r2_swin = stats.pearsonr(cc_vox_swin[~gtab.b0s_mask], csd_cc_SWIN[~gtab.b0s_mask])[0]**2

    cso_csd_r2 = stats.pearsonr(cso_vox[~gtab.b0s_mask], csd_cso[~gtab.b0s_mask])[0]**2
    cso_csd_r2_mp = stats.pearsonr(cso_vox_mp[~gtab.b0s_mask], csd_cso_MP[~gtab.b0s_mask])[0]**2
    cso_csd_r2_p2s = stats.pearsonr(cso_vox_p2s[~gtab.b0s_mask], csd_cso_P2S[~gtab.b0s_mask])[0]**2
    cso_csd_r2_swin = stats.pearsonr(cso_vox_swin[~gtab.b0s_mask], csd_cso_SWIN[~gtab.b0s_mask])[0]**2

    print(f"Corpus callosum\n CSD Raw R2: {cc_csd_r2}\n CSD MPPCA R2: {cc_csd_r2_mp}\n CSD P2S R2: {cc_csd_r2_p2s}\n CSD SWIN R2: {cc_csd_r2_swin}\n")
    print(f"Centrum Semiovale\n CSD Raw R2: {cso_csd_r2}\n CSD MPPCA R2: {cso_csd_r2_mp}\n CSD P2S R2: {cso_csd_r2_p2s}\n CSD SWIN R2: {cso_csd_r2_swin}\n")

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches([15, 7])
    ax[0].plot(cc_vox[~gtab.b0s_mask], csd_cc[~gtab.b0s_mask], 'X', color='crimson', label='Original: $R^{2}=' + f'{cc_csd_r2:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[0].plot(cc_vox_mp[~gtab.b0s_mask], csd_cc_MP[~gtab.b0s_mask], 'o', color='darkorange', label='Marchencko Pastur: $R^{2}=' + f'{cc_csd_r2_mp:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[0].plot(cc_vox_p2s[~gtab.b0s_mask], csd_cc_P2S[~gtab.b0s_mask], 'D', color='teal', label='Patch2Self: $R^{2}=' + f'{cc_csd_r2_p2s:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[0].plot(cc_vox_swin[~gtab.b0s_mask], csd_cc_SWIN[~gtab.b0s_mask], 's', color='darkviolet', label='SWIN: $R^{2}=' + f'{cc_csd_r2_swin:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')

    ax[1].plot(cso_vox[~gtab.b0s_mask], csd_cso[~gtab.b0s_mask], 'X', color='crimson', label='Original: $R^{2}=' + f'{cso_csd_r2:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[1].plot(cso_vox_mp[~gtab.b0s_mask], csd_cso_MP[~gtab.b0s_mask], 'o', color='darkorange', label='Marchencko Pastur: $R^{2}=' + f'{cso_csd_r2_mp:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[1].plot(cso_vox_p2s[~gtab.b0s_mask], csd_cso_P2S[~gtab.b0s_mask], 'D', color='teal', label='Patch2Self: $R^{2}=' + f'{cso_csd_r2_p2s:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')
    ax[1].plot(cso_vox_swin[~gtab.b0s_mask], csd_cso_SWIN[~gtab.b0s_mask], 's', color='darkviolet', label='SWIN: $R^{2}=' + f'{cso_csd_r2_swin:.3f}$', alpha=0.8, markeredgewidth=0.0, fillstyle='full')

    ax[0].set_title('Corpus Callosum')
    ax[1].set_title('Centrum Semiovale')

    for this_ax in ax:
        this_ax.set_xlabel('Data')
        this_ax.set_ylabel('Model prediction')
        this_ax.legend(loc='upper left', fontsize='large')
    fig.suptitle("CSD Cross Validation")
    fig.savefig("dmri-swin/figs/csd_cross_val.png", dpi=300, bbox_inches='tight')