"""
harris_stitcher.py

Author: Luke Griffin 21334528, Patrick Crotty 21336113, Michael Cronin 21329001, Aaron Smith 21335168,
    Cullen Toal 21306133
Date: 18-04-2025

Description:
Contains all core functions for Harris corner detection
"""

import numpy as np
from PIL import Image


def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def gaussian_derivative_kernels(sigma, size=5):
    ax = np.linspace(-(size // 2), size // 2, size)
    g = np.exp(-ax**2 / (2 * sigma**2))
    dg = -ax * g / sigma**2
    g /= g.sum()
    gx = np.outer(g, dg)  # ∂G/∂x
    gy = np.outer(dg, g)  # ∂G/∂y
    return gx, gy


def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output


def harris_response(image, sigma=1.0):
    gx, gy = gaussian_derivative_kernels(sigma, size=5)
    Ix = convolve2d(image, gx)
    Iy = convolve2d(image, gy)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    g_kernel = gaussian_kernel(size=5, sigma=2.5 * sigma)
    A = convolve2d(Ix2, g_kernel)
    B = convolve2d(Ixy, g_kernel)
    C = convolve2d(Iy2, g_kernel)
    det = A * C - B ** 2
    trace = A + C
    R = det / (trace + 1e-12)
    return R


def get_harris_points(harris_im, threshold=0.1, min_d=10):
    corner_threshold = harris_im.max() * threshold
    harris_im_th = (harris_im > corner_threshold)
    coords = np.array(harris_im_th.nonzero()).T
    candidate_values = np.array([harris_im[c[0], c[1]] for c in coords])
    indices = np.argsort(candidate_values)

    allowed_locations = np.zeros(harris_im.shape, dtype=bool)
    allowed_locations[min_d:-min_d, min_d:-min_d] = True
    filtered_coords = []
    for i in indices[::-1]:
        r, c = coords[i]
        if allowed_locations[r, c]:
            filtered_coords.append((r, c))
            allowed_locations[r-min_d:r+min_d+1, c-min_d:c+min_d+1] = False
    return filtered_coords


def get_descriptors(image, filtered_coords, wid=5):
    descriptors = []
    for r, c in filtered_coords:
        if r-wid >= 0 and c-wid >= 0 and r+wid+1 <= image.shape[0] and c+wid+1 <= image.shape[1]:
            patch = image[r-wid:r+wid+1, c-wid:c+wid+1].flatten()
            patch = patch - np.mean(patch)
            norm = np.linalg.norm(patch)
            if norm > 0:
                patch /= norm
            descriptors.append(patch)
        else:
            descriptors.append(np.zeros((2*wid+1)**2))  # fallback
    return np.array(descriptors)

def match_descriptors(desc1, desc2, threshold=0.95):
    R = np.dot(desc1, desc2.T)
    matches = []
    for i in range(R.shape[0]):
        j = np.argmax(R[i])
        if R[i, j] > threshold:
            matches.append((i, j))
    return matches


def exhaustive_ransac(matches, coords1, coords2, error_thresh=1.6):
    translations = []
    for i, j in matches:
        r1, c1 = coords1[i]
        r2, c2 = coords2[j]
        translations.append((r1 - r2, c1 - c2))

    best_translation = None
    best_support = 0
    for tr in translations:
        support = 0
        for ti in translations:
            dist = np.sqrt((tr[0]-ti[0])**2 + (tr[1]-ti[1])**2)
            if dist <= error_thresh:
                support += 1
        if support > best_support:
            best_support = support
            best_translation = tr
    return best_translation


def stitch_images(im1, im2, translation):
    dr, dc = translation
    r1, c1 = im1.shape
    r2, c2 = im2.shape

    if dr >= 0 and dc >= 0:
        out = np.zeros((dr + r2, dc + c2))
        out[:r1, :c1] = im1
        out[dr:dr+r2, dc:dc+c2] = im2
    elif dr < 0 and dc >= 0:
        out = np.zeros((-dr + r1, dc + c2))
        out[-dr:, :c1] = im1
        out[:r2, dc:dc+c2] = im2
    elif dr >= 0 and dc < 0:
        out = np.zeros((dr + r2, -dc + c1))
        out[:r1, -dc:] = im1
        out[dr:dr+r2, :c2] = im2
    else:
        out = np.zeros((-dr + r1, -dc + c1))
        out[-dr:, -dc:] = im1
        out[:r2, :c2] = im2
    return out


def load_image_grayscale(path):
    return np.array(Image.open(path).convert('L')) / 255.0
