"""
Generates synthetic OHRC-like and IIRS-like lunar terrain images
that represent the SAME underlying terrain but look visually different
(different resolution, contrast, noise, color mapping, crop offset).
Run once to populate data/ohrc.png and data/iirs.png.
"""
import numpy as np
import cv2


def make_terrain(size=900, seed=7):
    rng = np.random.default_rng(seed)
    h = np.zeros((size, size), dtype=np.float32)

    # Layer several octaves of smooth noise to build a heightmap (stronger,
    # more distinctive low-frequency texture so terrain patches are unique)
    for octave in range(6):
        scale = 2 ** octave
        small = rng.normal(0, 1, (scale * 3 + 2, scale * 3 + 2)).astype(np.float32)
        layer = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        h += 1.6 * layer / (scale ** 0.9)

    h = cv2.GaussianBlur(h, (0, 0), sigmaX=2)

    # A handful of distinctive large craters (unique sizes/positions reduce
    # SIFT ambiguity vs. many near-identical small craters)
    sizes = [85, 65, 50, 38, 30, 30, 22, 22, 18, 16, 14, 60, 45]
    for r in sizes:
        cx = rng.integers(50, size - 50)
        cy = rng.integers(50, size - 50)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bowl = -np.exp(-((dist / r) ** 2)) * (r / 18)
        rim = np.exp(-(((dist - r * 0.95) / (r * 0.22)) ** 2)) * (r / 32)
        h += (bowl + rim).astype(np.float32)

    # sprinkle small boulders/rocks for extra unique high-frequency texture
    for _ in range(60):
        cx = rng.integers(10, size - 10)
        cy = rng.integers(10, size - 10)
        r = rng.integers(3, 8)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bump = np.exp(-((dist / r) ** 2)) * (r / 25)
        h += bump.astype(np.float32)

    h = (h - h.min()) / (h.max() - h.min())
    return h  # normalized heightmap [0,1], represents real terrain


def height_to_ohrc(h):
    """OHRC: very high-res panchromatic, sharp, low sun angle -> strong shadows."""
    size = h.shape[0]
    # simulate shading from a low sun angle (grazing light -> strong shadow contrast)
    gy, gx = np.gradient(h)
    sun_dir = np.array([0.7, 0.7])
    shade = gx * sun_dir[0] + gy * sun_dir[1]
    shade = -shade
    shade = (shade - shade.min()) / (shade.max() - shade.min())
    img = 0.25 * h + 0.75 * shade
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.equalizeHist(img)
    # sharpen for "high resolution" panchromatic feel
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)
    noise = np.random.default_rng(1).normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def height_to_iirs(h):
    """IIRS: lower-res hyperspectral-like, high sun angle -> flatter shading,
    different color/intensity mapping, coarser detail (but craters still resolvable)."""
    size = h.shape[0]
    gy, gx = np.gradient(h)
    # different (but not opposite) sun angle than OHRC -> shading differs
    # noticeably in strength/softness but crater edges are not inverted,
    # which keeps cross-sensor matching physically plausible.
    sun_dir = np.array([0.5, 0.85])
    shade = gx * sun_dir[0] + gy * sun_dir[1]
    shade = -shade
    shade = (shade - shade.min()) / (shade.max() - shade.min())
    img = 0.6 * h + 0.4 * shade
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    # mild resolution loss (simulate coarser instrument, keep craters resolvable)
    small = cv2.resize(img, (int(size / 1.6), int(size / 1.6)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    # different contrast curve (gamma) to mimic different sensor response
    img_f = img.astype(np.float32) / 255.0
    img_f = np.power(img_f, 0.75)
    img = (img_f * 255).astype(np.uint8)
    noise = np.random.default_rng(2).normal(0, 3, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    full = make_terrain(size=900, seed=7)

    # OHRC: full extent, native crop
    ohrc_h = full[50:850, 50:850]
    ohrc = height_to_ohrc(ohrc_h)

    # IIRS: same physical terrain crop, slightly different footprint (a bit
    # wider margin, simulating a different instrument swath), then resampled
    # to a different pixel scale + different sensor look
    iirs_h = full[20:880, 20:880]
    iirs_h = cv2.resize(iirs_h, (700, 700))
    iirs = height_to_iirs(iirs_h)

    cv2.imwrite("data/ohrc.png", ohrc)
    cv2.imwrite("data/iirs.png", iirs)
    print("Generated data/ohrc.png", ohrc.shape)
    print("Generated data/iirs.png", iirs.shape)


if __name__ == "__main__":
    main()
