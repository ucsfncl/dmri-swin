import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
from dipy.io import image

from utilities import instantiate_from_config
from inference import swin_denoise, swin_denoise_low_mem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with swin")
    parser.add_argument('--dwi', help="Path to DWI file", required=True, type=str)
    parser.add_argument('--bvals', help="Path to bvals file", required=True, type=str)
    parser.add_argument('--mask', help="Path to mask file", required=True, type=str)
    parser.add_argument('--t1', help="Path to T1 file", required=True, type=str)
    parser.add_argument('--config', help="Path to config file", required=False, type=str, default="dmri-swin/swin_denoise.yaml")
    parser.add_argument('--output', help="Path to output file", required=False, type=str, default="swin_denoised.nii.gz")
    parser.add_argument('--resample', help="Resample to 1.25 mm", required=False, type=bool, default=True)
    parser.add_argument('--resample_back', help="Resample to dmri native resolution", required=False, type=bool, default=True)
    parser.add_argument('--low_mem', help="Use low memory version", required=False, type=bool, default=False)

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(config.ckpt_path, map_location="cpu"))

    t1, t1_affine = image.load_nifti(args.t1)
    dwi, dwi_affine = image.load_nifti(args.dwi)
    bvals = np.loadtxt(args.bvals)
    mask, _ = image.load_nifti(args.mask)

    if args.low_mem:
        dwi_swin, swin_affine = swin_denoise_low_mem(model, t1, dwi, bvals, mask, dwi_affine, t1_affine, resample=args.resample, resample_back=args.resample_back)
    else:
        dwi_swin, swin_affine = swin_denoise(model, t1, dwi, bvals, mask, dwi_affine, t1_affine, resample=args.resample, resample_back=args.resample_back)
    
    image.save_nifti(args.output, dwi_swin, swin_affine)
