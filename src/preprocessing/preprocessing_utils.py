from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class CTVolume:
    """A loaded CT volume with the metadata needed to map world <-> voxel coords."""
    array: np.ndarray       # shape (Z, Y, X), raw Hounsfield Units (HU)
    origin: np.ndarray      # world coords (mm) of voxel (0,0,0), order (x,y,z)
    spacing: np.ndarray     # voxel spacing (mm), order (x,y,z)

def load_mhd_image(path: str) -> CTVolume:
    """Load a LUNA16 .mhd/.raw pair into a CTVolume."""
    itk_image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk_image)  # (Z, Y, X), already in HU for LUNA16
    origin = np.array(itk_image.GetOrigin())      # (x, y, z)
    spacing = np.array(itk_image.GetSpacing())     # (x, y, z)
    return CTVolume(array=array, origin=origin, spacing=spacing)

def world_to_voxel(world_coord_xyz: np.ndarray, origin_xyz: np.ndarray,
                    spacing_xyz: np.ndarray) -> np.ndarray:
    """Convert a world (mm) coordinate (x, y, z) to a voxel index (x, y, z).

    Note: annotations.csv gives coords as (x, y, z); the loaded array is
    indexed as (z, y, x), so callers must reverse the order after this call.
    """
    return np.round((world_coord_xyz - origin_xyz) / spacing_xyz).astype(int)

def resample_to_spacing(volume: CTVolume, new_spacing=(1.0, 1.0, 1.0)) -> CTVolume:
    """Resample a CTVolume to isotropic voxel spacing via linear interpolation.

    Different LUNA16 scans have different slice thickness / pixel spacing.
    Standardizing this is necessary before patches from different patients
    can be compared or batched.
    """
    old_spacing_xyz = volume.spacing  # (x, y, z)
    old_shape_zyx = np.array(volume.array.shape)  # (z, y, x)

    # resize factor per axis, expressed in (z, y, x) to match array order
    old_spacing_zyx = old_spacing_xyz[::-1]
    new_spacing_zyx = np.array(new_spacing[::-1])
    resize_factor = old_spacing_zyx / new_spacing_zyx
    new_shape_zyx = np.round(old_shape_zyx * resize_factor).astype(int)

    from scipy.ndimage import zoom
    actual_resize_factor = new_shape_zyx / old_shape_zyx
    resampled_array = zoom(volume.array, actual_resize_factor, order=1)

    return CTVolume(array=resampled_array, origin=volume.origin,
                     spacing=np.array(new_spacing))

def apply_lung_window(hu_array: np.ndarray, window_center: int = -600,
                       window_width: int = 1500) -> np.ndarray:
    """Apply a lung HU window and normalize to [0, 1] float32.

    Default center/width (-600 / 1500) is a standard lung window covering
    roughly -1350 to 150 HU, which captures lung parenchyma and nodules
    while clipping bone/metal artifacts. Adjust in configs/ if ablations
    show a tighter nodule-focused window helps.
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    clipped = np.clip(hu_array, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return normalized.astype(np.float32)

def extract_patch(volume_array: np.ndarray, center_voxel_zyx, patch_size=(32, 64, 64)):
    """Extract a fixed-size 3D patch centered on a voxel, zero-padding at edges.

    center_voxel_zyx: (z, y, x) voxel index of the patch center.
    patch_size: (z, y, x) size of the output patch.
    """
    center = np.array(center_voxel_zyx)
    half = np.array(patch_size) // 2

    start = center - half
    end = start + np.array(patch_size)

    pad_before = np.clip(-start, 0, None)
    pad_after = np.clip(end - np.array(volume_array.shape), 0, None)

    clipped_start = np.clip(start, 0, None)
    clipped_end = np.clip(end, None, np.array(volume_array.shape))

    patch = volume_array[
        clipped_start[0]:clipped_end[0],
        clipped_start[1]:clipped_end[1],
        clipped_start[2]:clipped_end[2],
    ]

    pad_width = list(zip(pad_before, pad_after))
    patch = np.pad(patch, pad_width, mode="constant", constant_values=0)

    return patch


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python preprocessing_utils.py <path_to.mhd>")
        sys.exit(1)

    vol = load_mhd_image(sys.argv[1])
    print(f"Loaded volume: shape={vol.array.shape}, spacing={vol.spacing}, "
          f"origin={vol.origin}")
    print(f"HU range: [{vol.array.min():.1f}, {vol.array.max():.1f}]")

    resampled = resample_to_spacing(vol)
    print(f"Resampled shape: {resampled.array.shape}")

    windowed = apply_lung_window(resampled.array)
    print(f"Windowed range: [{windowed.min():.3f}, {windowed.max():.3f}]")