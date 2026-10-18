# -*- coding: utf-8 -*-
"""
PyTorch (GPU) variants of the SAM feature functions of SPOT.Features.features.

compute_boundary_morphology_features_torch, compute_boundary_texture_features_torch and
compute_boundary_motion_features_torch take the same arguments as the original functions plus ``device`` and
``batch_size``, and return the same outputs. The heavy kernels are batched over all (object, frame) samples on the
device:

    - polygon rasterisation (contour masks and regional windows), reproducing skimage.draw.polygon exactly
    - grey-level co-occurrence matrices and the 13 Haralick features (as mahotas.features.haralick)
    - the global SIFT descriptor of a patch (as SingleSiftExtractor)
    - the Euler characteristic transform histograms
    - curl and divergence of the optical flow

The rest (spline curvature, regionprops / Zernike moments, masked SIFT and the distance-halving fallback of the masked
Haralick features, the speed density estimates) runs on the CPU exactly as in the original functions. Results match the
original functions up to floating point round-off of the batched reductions (computed in float64).

torch is imported lazily, so SPOT does not require it unless these functions are used.

@author: fyz11
"""

import numpy as np

from . import features as sf


def _torch():
    import torch
    return torch


def _get_device(device):
    torch = _torch()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


# =============================================================================
#   Polygon rasterisation
# =============================================================================
def polygon_torch(polygons, shapes=None, device=None, max_elements=2**23):
    """ Rasterise many polygons on the device, giving exactly skimage.draw.polygon(r, c, shape) for each.

    skimage tests every pixel of the bounding box with left / right ray crossing counts (pixels on an edge or vertex are
    inside). For a given row and edge the crossing test is monotone in the column except within ~1e-13 pixel of the
    exact crossing, so it is evaluated with skimage's expression only at the 4 columns around the crossing and taken as
    constant on either side; the crossing parities of all edges are then accumulated with prefix sums. The cost is
    O(n_edges * n_rows) per polygon instead of O(n_edges * n_rows * n_cols).

    Parameters
    ----------
    polygons : list of (r, c) tuples of 1d float arrays
        the row and column coordinates of the vertices of each polygon
    shapes : None or list of None or (n_rows, n_cols)
        the image shape clipping each polygon, as the shape argument of skimage.draw.polygon
    device : None or str or torch.device
        the device to use. None uses cuda if available.
    max_elements : int
        the maximum number of (polygon, edge, row) elements processed at once

    Returns
    -------
    out : list of (rr, cc) tuples of 1d int arrays
        the pixels inside each polygon in row-major order, as returned by skimage.draw.polygon
    """
    torch = _torch()
    device = _get_device(device)
    n_poly = len(polygons)
    if shapes is None:
        shapes = [None] * n_poly
    eps = 1e-12

    # bounding boxes, as skimage.draw._polygon
    boxes = []
    for (r, c), shape in zip(polygons, shapes):
        r = np.asarray(r, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
        minr = int(max(0, r.min())); maxr = int(np.ceil(r.max()))
        minc = int(max(0, c.min())); maxc = int(np.ceil(c.max()))
        if shape is not None:
            maxr = min(shape[0] - 1, maxr)
            maxc = min(shape[1] - 1, maxc)
        boxes.append((minr, maxr, minc, maxc))

    out = [None] * n_poly
    empty = (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp))
    valid_all = [i for i in range(n_poly) if boxes[i][1] >= boxes[i][0] and boxes[i][3] >= boxes[i][2]]
    valid_set = set(valid_all)
    for i in range(n_poly):
        if i not in valid_set:
            out[i] = empty

    # batches of polygons with similar sizes
    order = sorted(valid_all, key=lambda i: (boxes[i][1] - boxes[i][0], boxes[i][3] - boxes[i][2]))
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order):
            idx = order[pos:end + 1]
            n_e = max(len(polygons[i][0]) for i in idx)
            h = max(boxes[i][1] - boxes[i][0] + 1 for i in idx)
            w = max(boxes[i][3] - boxes[i][2] + 1 for i in idx)
            if len(idx) * h * max(n_e * 4, w + 1) > max_elements:
                break
            end += 1
        valid = order[pos:end]
        pos = end

        B = len(valid)
        n_verts = max(len(polygons[i][0]) for i in valid)
        H = max(boxes[i][1] - boxes[i][0] + 1 for i in valid)
        W = max(boxes[i][3] - boxes[i][2] + 1 for i in valid)
        # pad with copies of the last vertex: zero length edges never straddle a row
        R = np.zeros((B, n_verts)); C = np.zeros((B, n_verts))
        for k, i in enumerate(valid):
            r = np.asarray(polygons[i][0], dtype=np.float64); c = np.asarray(polygons[i][1], dtype=np.float64)
            R[k, :len(r)] = r; R[k, len(r):] = r[-1]
            C[k, :len(c)] = c; C[k, len(c):] = c[-1]
        yp = torch.from_numpy(R).to(device); xp = torch.from_numpy(C).to(device)
        r0 = torch.tensor([boxes[i][0] for i in valid], dtype=torch.float64, device=device)
        c0 = torch.tensor([boxes[i][2] for i in valid], dtype=torch.int64, device=device)
        y = r0[:, None] + torch.arange(H, dtype=torch.float64, device=device)[None, :]  # (B, H)

        # edge e joins vertex e-1 (j) to vertex e (i), as in the skimage loop
        xi = xp[:, :, None]; xj = torch.roll(xp, 1, dims=1)[:, :, None]  # (B, E, 1)
        y0 = yp[:, :, None] - y[:, None, :]  # (B, E, H)  yp[i] - y
        y1 = torch.roll(yp, 1, dims=1)[:, :, None] - y[:, None, :]  # yp[j] - y
        straddle_r = (y0 > 0) != (y1 > 0)
        straddle_l = (y0 < 0) != (y1 < 0)
        den = y1 - y0
        sgn = torch.sign(den)
        # estimated crossing column (exact up to ~1e-13 pixel)
        safe_den = torch.where(den == 0, torch.ones_like(den), den)
        xs = torch.where(straddle_r | straddle_l, (xi * y1 - xj * y0) / safe_den, torch.zeros_like(den))
        c0f = c0[:, None, None].to(torch.float64)
        base = torch.floor(torch.minimum(torch.maximum(xs, c0f - 4), c0f + W + 4)).to(torch.int64)  # (B, E, H)

        rows = torch.arange(H, device=device)[None, None, :].expand(B, n_verts, H)
        bidx = torch.arange(B, device=device)[:, None, None].expand(B, n_verts, H)
        c0b = c0[:, None, None]
        # right crossings: test true for all columns < base - 1 -> prefix toggles, counted by a suffix sum
        P = torch.zeros(B * H * (W + 1), dtype=torch.int32, device=device)
        a_rel = torch.clamp(base - 1 - c0b, 0, W)
        P.index_add_(0, ((bidx * H + rows) * (W + 1) + a_rel)[straddle_r],
                     torch.ones(int(straddle_r.sum()), dtype=torch.int32, device=device))
        # left crossings: test true for all columns > base + 2 -> suffix toggles, counted by a prefix sum
        Q = torch.zeros(B * H * (W + 1), dtype=torch.int32, device=device)
        s_rel = torch.clamp(base + 3 - c0b, 0, W)
        Q.index_add_(0, ((bidx * H + rows) * (W + 1) + s_rel)[straddle_l],
                     torch.ones(int(straddle_l.sum()), dtype=torch.int32, device=device))
        P = P.view(B, H, W + 1); Q = Q.view(B, H, W + 1)
        r_count = torch.flip(torch.cumsum(torch.flip(P, dims=[2]), dim=2), dims=[2])[:, :, 1:]  # edges with a_rel > col
        l_count = torch.cumsum(Q, dim=2)[:, :, :W]  # edges with s_rel <= col

        # exact skimage test at the 4 candidate columns base-1 .. base+2
        CR = torch.zeros(B * H * W, dtype=torch.int32, device=device)
        CL = torch.zeros(B * H * W, dtype=torch.int32, device=device)
        for off in (-1, 0, 1, 2):
            xc_int = base + off
            rel = xc_int - c0b
            inb = (rel >= 0) & (rel < W)
            xc = xc_int.to(torch.float64)
            x0 = xi - xc
            x1 = xj - xc
            signed_num = (x0 * y1 - x1 * y0) * sgn
            flat = (bidx * H + rows) * W + torch.clamp(rel, 0, W - 1)
            sel_r = straddle_r & inb & (signed_num > 0)
            sel_l = straddle_l & inb & (signed_num < 0)
            CR.index_add_(0, flat[sel_r], torch.ones(int(sel_r.sum()), dtype=torch.int32, device=device))
            CL.index_add_(0, flat[sel_l], torch.ones(int(sel_l.sum()), dtype=torch.int32, device=device))
        r_par = ((r_count + CR.view(B, H, W)) & 1).bool()
        l_par = ((l_count + CL.view(B, H, W)) & 1).bool()
        inside_np = (r_par | (r_par != l_par)).cpu().numpy()

        for k, i in enumerate(valid):
            h = boxes[i][1] - boxes[i][0] + 1; w = boxes[i][3] - boxes[i][2] + 1
            ins = inside_np[k, :h, :w]
            # pixels on a vertex (within eps) are inside
            r = np.asarray(polygons[i][0], dtype=np.float64); c = np.asarray(polygons[i][1], dtype=np.float64)
            ri, ci = np.round(r), np.round(c)
            on_vertex = (np.abs(r - ri) < eps) & (np.abs(c - ci) < eps) & (ri >= boxes[i][0]) & (ri <= boxes[i][1]) \
                        & (ci >= boxes[i][2]) & (ci <= boxes[i][3])
            if on_vertex.any():
                ins = ins.copy()
                ins[(ri[on_vertex] - boxes[i][0]).astype(np.intp), (ci[on_vertex] - boxes[i][2]).astype(np.intp)] = True
            rr, cc = np.nonzero(ins)
            out[i] = (rr + boxes[i][0], cc + boxes[i][2])
    return out


def contour_to_binary_torch(contours, device=None):
    """ contour_to_binary(cnt, shape=None, return_bbox=True) for many contours, rasterised on the device.

    Returns
    -------
    out : list of (binary, bbox)
        identical to SPOT.Features.features.contour_to_binary(cnt, shape=None, return_bbox=True)
    """
    polys, shapes, meta = [], [], []
    for cnt in contours:
        min_x = int(np.min(cnt[:,1])); max_x = int(np.max(cnt[:,1]))
        min_y = int(np.min(cnt[:,0])); max_y = int(np.max(cnt[:,0]))
        m = max_y - min_y + 1; n = max_x - min_x + 1
        polys.append((cnt[:,0] - min_y, cnt[:,1] - min_x)); shapes.append((m, n))
        meta.append((m, n, np.hstack([min_x, min_y, max_x, max_y])))
    pix = polygon_torch(polys, shapes, device=device)
    out = []
    for (rr, cc), (m, n, bbox) in zip(pix, meta):
        blank = np.zeros((m, n))
        blank[rr, cc] = 1
        out.append((blank > .5, bbox))
    return out


def radial_windows_torch(imsize, n_r, n_theta, boundary_lines, zero_angle=None, device=None):
    """ _tile_uniform_radial_windows_boundary_crop for many boundaries with the window polygons rasterised on the device.

    Returns
    -------
    out : list of (spixels, (r0, c0))
        identical to SPOT.Features.features._tile_uniform_radial_windows_boundary_crop
    """
    polys, counts = [], []
    for line in boundary_lines:
        int_lines, _ = sf._radial_window_int_lines(imsize, n_r, line)
        polys += [(yy, xx) for yy, xx in int_lines]
        counts.append(len(int_lines))
    pix = polygon_torch(polys, None, device=device)  # _draw_polygon_mask rasterises without a shape
    out, pos = [], 0
    for line, k in zip(boundary_lines, counts):
        out.append(sf._tile_uniform_radial_windows_boundary_crop(imsize, n_r, n_theta, line, zero_angle=zero_angle,
                                                                 line_pixels=pix[pos:pos + k]))
        pos += k
    return out


# =============================================================================
#   Haralick features
# =============================================================================
_2D_DELTAS = [(0, 1), (1, 1), (1, 0), (1, -1)]  # mahotas.features.texture._2d_deltas


def haralick_torch(images, distances, ignore_zeros=False, device=None, max_bytes=2**28):
    """ mahotas.features.haralick(f, ignore_zeros, preserve_haralick_bug=False, compute_14th_feature=False,
    return_mean=True, return_mean_ptp=False, use_x_minus_y_variance=False, distance=d) for many integer images.

    Parameters
    ----------
    images : list of 2d non-negative integer arrays
    distances : list of int
        the co-occurrence distance of each image
    ignore_zeros : bool
        as mahotas: pixel pairs involving grey level 0 are not counted
    device : None or str or torch.device
    max_bytes : int
        approximate memory for the co-occurrence matrices of one batch

    Returns
    -------
    feats : (n_images, 13) float64 array
        the mean over the 4 directions of the 13 Haralick features
    empty : (n_images,) bool array
        True where a co-occurrence matrix is empty, for which mahotas raises a ValueError (feats are NaN there)
    """
    torch = _torch()
    device = _get_device(device)
    n = len(images)
    feats = np.full((n, 13), np.nan)
    empty = np.zeros(n, dtype=bool)
    levels = np.array([int(np.max(im)) + 1 for im in images])
    order = np.argsort(levels, kind='stable')
    pos = 0
    while pos < n:
        L = int(levels[order[pos]])
        size = max(1, int(max_bytes // (8 * 4 * L * L * 4)))
        idx = order[pos:pos + size]
        L = int(levels[idx].max())
        pos += len(idx)
        B = len(idx)

        # co-occurrence matrices (B, 4, L, L), symmetric, as mahotas cooccurence
        cmat = torch.zeros(B * 4 * L * L, dtype=torch.float64, device=device)
        for k, i in enumerate(idx):
            f = torch.from_numpy(np.ascontiguousarray(images[i]).astype(np.int64)).to(device)
            h, w = f.shape
            d = int(distances[i])
            for di, (dy, dx) in enumerate(_2D_DELTAS):
                dy, dx = dy * d, dx * d
                # pairs (f[y, x], f[y+dy, x+dx]) with both pixels in the image
                a = f[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
                b = f[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
                if a.numel() == 0:
                    continue
                base = (k * 4 + di) * L * L
                ones = torch.ones(a.numel(), dtype=torch.float64, device=device)
                cmat.index_add_(0, (base + a * L + b).ravel(), ones)
                cmat.index_add_(0, (base + b * L + a).ravel(), ones)
        cmat = cmat.view(B, 4, L, L)
        if ignore_zeros:
            cmat[:, :, 0, :] = 0
            cmat[:, :, :, 0] = 0
        T = cmat.sum(dim=(2, 3))  # (B, 4)
        is_empty = (T == 0).any(dim=1)
        T = torch.where(T == 0, torch.ones_like(T), T)
        p = cmat / T[:, :, None, None]

        k_ = torch.arange(L, dtype=torch.float64, device=device)
        k2 = k_**2
        tk = torch.arange(2 * L, dtype=torch.float64, device=device)
        tk2 = tk**2
        ii, jj = torch.meshgrid(k_, k_, indexing='ij')
        ij = ii * jj
        i_j2_p1 = 1. / ((ii - jj)**2 + 1)
        plus_idx = (ii + jj).long().ravel()
        minus_idx = (ii - jj).abs().long().ravel()

        def entropy(q):
            q = q.reshape(q.shape[0], q.shape[1], -1)
            return -(torch.log2(q + (q == 0).to(q.dtype)) * q).sum(-1)

        px = p.sum(2)  # p.sum(0) in mahotas: sum over rows
        py = p.sum(3)
        ux = (px * k_).sum(-1); uy = (py * k_).sum(-1)
        vx = (px * k2).sum(-1) - ux**2
        vy = (py * k2).sum(-1) - uy**2
        sx = torch.sqrt(vx); sy = torch.sqrt(vy)
        pflat = p.reshape(B, 4, L * L)
        px_plus_y = torch.zeros(B, 4, 2 * L, dtype=torch.float64, device=device).index_add_(2, plus_idx, pflat)
        px_minus_y = torch.zeros(B, 4, L, dtype=torch.float64, device=device).index_add_(2, minus_idx, pflat)

        f = torch.zeros(B, 4, 13, dtype=torch.float64, device=device)
        f[..., 0] = (pflat * pflat).sum(-1)
        f[..., 1] = (px_minus_y * k2).sum(-1)
        corr = (1. / sx / sy) * ((pflat * ij.ravel()).sum(-1) - ux * uy)
        f[..., 2] = torch.where((sx == 0) | (sy == 0), torch.ones_like(corr), corr)
        f[..., 3] = vx
        f[..., 4] = (pflat * i_j2_p1.ravel()).sum(-1)
        f[..., 5] = (px_plus_y * tk).sum(-1)
        f[..., 7] = entropy(px_plus_y)
        f[..., 6] = (px_plus_y * tk2).sum(-1) - f[..., 5]**2
        f[..., 8] = entropy(pflat)
        # px_minus_y.var() over the image's own number of grey levels
        Ls = torch.from_numpy(levels[idx].astype(np.float64)).to(device)[:, None]
        valid_k = (k_[None, None, :] < Ls[:, :, None]).to(torch.float64)
        mean_pmy = px_minus_y.sum(-1) / Ls
        f[..., 9] = (((px_minus_y - mean_pmy[..., None])**2) * valid_k).sum(-1) / Ls
        f[..., 10] = entropy(px_minus_y)
        HX = entropy(px); HY = entropy(py)
        cross = px[..., :, None] * py[..., None, :]
        cross = cross + (cross == 0).to(cross.dtype)
        cross = cross.reshape(B, 4, L * L)
        HXY1 = -(pflat * torch.log2(cross)).sum(-1)
        HXY2 = -(torch.log2(cross) * cross).sum(-1)
        hmax = torch.maximum(HX, HY)
        safe_hmax = torch.where(hmax == 0, torch.ones_like(hmax), hmax)
        f[..., 11] = torch.where(hmax == 0, f[..., 8] - HXY1, (f[..., 8] - HXY1) / safe_hmax)
        f[..., 12] = torch.sqrt(torch.clamp(1 - torch.exp(-2. * (HXY2 - f[..., 8])), min=0))

        res = f.mean(dim=1).cpu().numpy()
        is_empty = is_empty.cpu().numpy()
        res[is_empty] = np.nan
        feats[idx] = res
        empty[idx] = is_empty
    return feats, empty


# =============================================================================
#   SIFT descriptor of a whole patch
# =============================================================================
def sift_torch(images, device=None, batch_size=1024):
    """ SingleSiftExtractor(patch_size).process_image(im).ravel() for many square images of the same size.

    Parameters
    ----------
    images : (n_images, size, size) array
        the (already resized) patches
    device : None or str or torch.device
    batch_size : int

    Returns
    -------
    feats : (n_images, 128) float64 array
        the SIFT descriptors before the final L2 normalisation of the feature functions
    """
    torch = _torch()
    import torch.nn.functional as F
    device = _get_device(device)
    images = np.asarray(images)
    n, size = images.shape[0], images.shape[1]
    ext = sf.SingleSiftExtractor(size)
    GH, GW = ext.gen_dgauss(ext.sigma)
    # scipy.signal.convolve2d(mode='same') is a correlation with the flipped kernel, zero padded
    kh = torch.from_numpy(np.ascontiguousarray(GH[::-1, ::-1])).to(device)[None, None]
    kw = torch.from_numpy(np.ascontiguousarray(GW[::-1, ::-1])).to(device)[None, None]
    pad = GH.shape[0] // 2
    weights = torch.from_numpy(ext.weights).to(device)  # (Nsamples, size*size)
    angles = torch.from_numpy(ext.angles).to(device)
    out = np.zeros((n, ext.Nangles * ext.Nsamples))
    for sl in _chunks(n, batch_size):
        # process_image casts to float32, convolve2d then works in float64
        im = torch.from_numpy(images[sl].astype(np.float32).astype(np.float64)).to(device)[:, None]
        IH = F.conv2d(im, kh, padding=pad)[:, 0]
        IW = F.conv2d(im, kw, padding=pad)[:, 0]
        Imag = torch.sqrt(IH**2 + IW**2)
        Itheta = torch.atan2(IH, IW)
        Iorient = Imag[:, None] * torch.clamp(torch.cos(Itheta[:, None] - angles[None, :, None, None])**ext.alpha, min=0)
        fea = torch.einsum('sp,bap->bas', weights, Iorient.reshape(Iorient.shape[0], ext.Nangles, -1))
        fea = fea.reshape(fea.shape[0], -1)
        # DsiftExtractor.normalize_sift
        siftlen = torch.sqrt((fea**2).sum(1))
        hcontrast = siftlen >= ext.nrml_thres
        siftlen = torch.clamp(siftlen, min=ext.nrml_thres)
        fea = fea / siftlen[:, None]
        fea = torch.clamp(fea, max=ext.sift_thres)
        renorm = torch.sqrt((fea**2).sum(1))
        fea = torch.where(hcontrast[:, None], fea / torch.where(renorm == 0, torch.ones_like(renorm), renorm)[:, None], fea)
        out[sl] = fea.cpu().numpy()
    return out


# =============================================================================
#   Euler characteristic transform
# =============================================================================
def _cubical_cells(binary):
    """ centred vertices, edges and squares of the cubical complex of a binary image, as CubicalComplex.complexify """
    on = np.asarray(binary) > 0
    coords = np.vstack(np.nonzero(on)).T
    verts = sf._centerVertices(coords)
    index = np.full(on.shape, -1, dtype=np.int64)
    index[on] = np.arange(len(coords))
    horz = on[:, :-1] & on[:, 1:]
    vert = on[:-1, :] & on[1:, :]
    edges = np.vstack([np.vstack([index[:, :-1][horz], index[:, 1:][horz]]).T,
                       np.vstack([index[:-1, :][vert], index[1:, :][vert]]).T])
    sq = on[:-1, :-1] & on[:-1, 1:] & on[1:, :-1] & on[1:, 1:]
    squares = np.vstack([index[:-1, :-1][sq], index[:-1, 1:][sq], index[1:, :-1][sq], index[1:, 1:][sq]]).T
    return verts, edges, squares


def ect_torch(binaries, directions, T=32, device=None, max_vertices=2**19):
    """ CubicalComplex(binary).complexify().ECT(directions, T) for many binary images.

    The bins are found as np.histogram does (bin edges from np.linspace over the [min, max] height range, the last bin
    closed), so the result is identical.

    Returns
    -------
    ect : (n_images, n_directions*T) int array
    """
    torch = _torch()
    device = _get_device(device)
    n = len(binaries)
    D = directions.shape[0]
    out = np.zeros((n, D * T), dtype=int)
    cells = [_cubical_cells(b) for b in binaries]
    dirs = torch.from_numpy(np.asarray(directions, dtype=np.float64)).to(device)
    pos = 0
    while pos < n:
        end, nv = pos, 0
        while end < n and (end == pos or nv + len(cells[end][0]) <= max_vertices):
            nv += len(cells[end][0]); end += 1
        batch = cells[pos:end]
        B = end - pos
        verts = torch.from_numpy(np.vstack([c[0] for c in batch])).to(device)
        vsample = torch.from_numpy(np.repeat(np.arange(B), [len(c[0]) for c in batch]).astype(np.int64)).to(device)
        offsets = np.cumsum([0] + [len(c[0]) for c in batch])[:-1]
        edges = torch.from_numpy(np.vstack([c[1] + o for c, o in zip(batch, offsets)] + [np.zeros((0, 2), int)]).astype(np.int64)).to(device)
        esample = torch.from_numpy(np.repeat(np.arange(B), [len(c[1]) for c in batch]).astype(np.int64)).to(device)
        squares = torch.from_numpy(np.vstack([c[2] + o for c, o in zip(batch, offsets)] + [np.zeros((0, 4), int)]).astype(np.int64)).to(device)
        ssample = torch.from_numpy(np.repeat(np.arange(B), [len(c[2]) for c in batch]).astype(np.int64)).to(device)

        # heights: same arithmetic as np.sum(verts*direction, axis=1)
        heights = (verts[:, None, :] * dirs[None, :, :]).sum(-1)  # (Nv, D)
        minh = torch.full((B, D), float('inf'), dtype=torch.float64, device=device).scatter_reduce(
            0, vsample[:, None].expand(-1, D), heights, reduce='amin')
        maxh = torch.full((B, D), -float('inf'), dtype=torch.float64, device=device).scatter_reduce(
            0, vsample[:, None].expand(-1, D), heights, reduce='amax')
        # np.histogram widens an empty range by 0.5 on each side
        same = minh == maxh
        first = torch.where(same, minh - 0.5, minh)
        last = torch.where(same, maxh + 0.5, maxh)
        # bin edges as np.linspace(first, last, T+1)
        step = (last - first) / torch.full_like(last, float(T))  # tensor division: CUDA divides by a scalar through its reciprocal
        edges_lin = torch.arange(T + 1, dtype=torch.float64, device=device)[None, None, :] * step[:, :, None] + first[:, :, None]
        edges_lin[:, :, -1] = last

        def bins(values, sample):
            # values (N, D) -> bin index (N, D) with edges[k] <= v < edges[k+1], last bin closed, found as np.histogram:
            # estimate from the bin width, then correct by one bin against the actual edges
            f0 = first[sample]; f1 = last[sample]
            k = torch.floor((values - f0) / (f1 - f0) * T).to(torch.int64).clamp(0, T - 1)
            row = (sample[:, None] * D + dcol) * (T + 1)  # (N, D) offsets into the flat (B, D, T+1) edges
            edges_flat = edges_lin.reshape(-1)
            lo = edges_flat[row + k]
            k = torch.where(values < lo, k - 1, k)
            hi = edges_flat[row + k + 1]
            k = torch.where((values >= hi) & (k != T - 1), k + 1, k)
            return k

        counts = torch.zeros(B * D * T, dtype=torch.int64, device=device)
        dcol = torch.arange(D, device=device)[None, :]

        def add(values, sample, sign):
            if values.shape[0] == 0:
                return
            k = bins(values, sample)
            flat = (sample[:, None] * D + dcol) * T + k
            counts.index_add_(0, flat.ravel(), torch.full((flat.numel(),), sign, dtype=torch.int64, device=device))

        add(heights, vsample, 1)
        if edges.shape[0] > 0:
            add(torch.maximum(heights[edges[:, 0]], heights[edges[:, 1]]), esample, -1)
        if squares.shape[0] > 0:
            hs = torch.maximum(torch.maximum(heights[squares[:, 0]], heights[squares[:, 1]]),
                               torch.maximum(heights[squares[:, 2]], heights[squares[:, 3]]))
            add(hs, ssample, 1)
        ecc = torch.cumsum(counts.view(B, D, T), dim=2)
        out[pos:end] = ecc.reshape(B, D * T).cpu().numpy()
        pos = end
    return out


# =============================================================================
#   curl and divergence
# =============================================================================
def _gradient_np_like(f, axis):
    """ np.gradient(f, axis=axis) with unit spacing and edge_order=1, with the same arithmetic. """
    torch = _torch()
    out = torch.empty_like(f)
    n = f.shape[axis]
    sl = lambda a, b: tuple(slice(a, b) if d == axis else slice(None) for d in range(f.dim()))
    out[sl(1, n - 1)] = (f[sl(2, n)] - f[sl(0, n - 2)]) / 2.
    out[sl(0, 1)] = (f[sl(1, 2)] - f[sl(0, 1)]) / 1.
    out[sl(n - 1, n)] = (f[sl(n - 1, n)] - f[sl(n - 2, n - 1)]) / 1.
    return out


def curl_div_torch(flow_frame, time_res=1., device=None):
    """ (_curl_vector_flow(flow)[...,-1] / time_res, _div_vector_flow(flow) / time_res) of a (n_rows, n_cols, 2) flow, computed on the device. Returns numpy arrays. """
    torch = _torch()
    device = _get_device(device)
    f = torch.from_numpy(np.ascontiguousarray(flow_frame)).to(device)
    Fx = f[..., 0]; Fy = f[..., 1]
    dFx_dy = _gradient_np_like(Fx, 0); dFx_dx = _gradient_np_like(Fx, 1)
    dFy_dy = _gradient_np_like(Fy, 0); dFy_dx = _gradient_np_like(Fy, 1)
    curl = (dFy_dx - dFx_dy).cpu().numpy()
    div = (dFx_dx + dFy_dy).cpu().numpy()
    # divide on the CPU: CUDA divides by a scalar through its reciprocal, which can differ in the last bit.
    # As the original, the curl is returned in float64 (np.dstack with float64 zeros) and the divergence in the flow dtype.
    return curl.astype(np.float64) / float(time_res), div / float(time_res)


# =============================================================================
#   SAM feature functions
# =============================================================================
def _valid_samples(boundaries, frames, need_next=False):
    """ (object, frame index, frame) of all samples with a contour (and a contour in the next frame if need_next) """
    out = []
    for b_ii in range(len(boundaries)):
        for jj, tt in enumerate(frames):
            if np.isnan(boundaries[b_ii][tt][0][0]):
                continue
            if need_next and np.isnan(boundaries[b_ii][tt+1][0][0]):
                continue
            out.append((b_ii, jj, tt))
    return out


def _reorient_from_pixels(cnt, rr, cc):
    """ reorient_contour_fast given the rasterised contour pixels """
    from skimage.measure import regionprops
    r0, c0 = rr.min(), cc.min()
    canvas = np.zeros((rr.max()-r0+1, cc.max()-c0+1), dtype=np.int32)
    canvas[rr-r0, cc-c0] = 1
    orientation = regionprops(canvas)[0].orientation
    center = cnt[:-1].mean(axis=0)
    cnt_rot = sf._rotate_pts(cnt[:-1,::-1], angle=orientation/np.pi*180., center=center[::-1])[:,::-1]
    cnt_angle = np.arctan2(cnt_rot[:, 0] - center[0], cnt_rot[:,1]- center[1])
    sort = np.argsort(cnt_angle)
    cnt_rot_ = np.vstack([cnt_rot[sort[0]:], cnt_rot[:sort[0]]])
    return np.vstack([cnt_rot_, cnt_rot_[0]])


def compute_boundary_morphology_features_torch(boundaries, imshape,
                                               all_feats_compute=True,
                                               contour_curve_feats=True,
                                               curve_order=4, curve_err=1.,
                                               geom_features=True,
                                               morph_features=True,
                                               fourier_features=True,
                                               ect_features=True,
                                               shape_context_features=True,
                                               norm_scm=True,
                                               n_ref_pts_scm=5,
                                               pixel_xy_res=1.,
                                               timesample=None,
                                               device=None,
                                               fourier_remove_closing_point=False,
                                               elliptic_fourier_features=True,
                                               elliptic_fourier_order=10):
    """ PyTorch variant of SPOT.Features.features.compute_boundary_morphology_features with the same parameters and outputs.

    The contour rasterisations (for the reorientation and the binary shape) and the Euler characteristic transform are
    batched over all samples on the device (identical results); the spline curvature, regionprops / Hu / Zernike
    moments, contour geometry, Fourier and shape context features are computed on the CPU as in
    compute_boundary_morphology_features_fast.

    Parameters
    ----------
    (see compute_boundary_morphology_features)
    device : None or str or torch.device
        the torch device. None uses cuda if available, else the CPU.
    fourier_remove_closing_point : bool
        if True, the Fourier features drop the repeated closing point of the contour, which makes them exactly invariant
        to rotation and to the contour start point (there is then one Fourier feature less). False (default) keeps the
        original features.

    Returns
    -------
    out_array : (n_organoids, n_frames, n_features) array
    out_feat_labels : (n_features,) str array
    out_feat_labels_norm : (n_features,) bool array
    """
    from tqdm import tqdm
    # C order: the features are sensitive at the last bit to the memory layout of the contours (e.g. reductions such as
    # the contour mean), which can flip ties on pixel-grid contours; a fixed layout makes all paths identical.
    boundaries = np.ascontiguousarray(boundaries)
    if timesample is None:
        timesample = 1
    if all_feats_compute==False and ect_features==True and morph_features==False:
        raise ValueError('ect_features requires morph_features=True, as in compute_boundary_morphology_features')
    frames = np.arange(0, boundaries.shape[1], timesample)
    opts = dict(all_feats_compute=all_feats_compute, contour_curve_feats=contour_curve_feats, curve_order=curve_order,
                curve_err=curve_err, geom_features=geom_features, morph_features=morph_features,
                fourier_features=fourier_features, ect_features=ect_features,
                shape_context_features=shape_context_features, norm_scm=norm_scm, n_ref_pts_scm=n_ref_pts_scm,
                pixel_xy_res=pixel_xy_res, fourier_remove_closing_point=fourier_remove_closing_point,
                elliptic_fourier_features=elliptic_fourier_features, elliptic_fourier_order=elliptic_fourier_order)
    samples = _valid_samples(boundaries, frames)
    contours = [boundaries[b][tt] for b, jj, tt in samples]

    # reorientation (rasterised in the image) and binary shapes on the device
    pix = polygon_torch([(c[:,0], c[:,1]) for c in contours], [tuple(imshape)] * len(contours), device=device)
    reoriented = [_reorient_from_pixels(c, rr, cc) for c, (rr, cc) in zip(contours, pix)]
    pre = [dict(reoriented=r) for r in reoriented]
    need_binary = all_feats_compute or morph_features
    if need_binary:
        binaries = contour_to_binary_torch(reoriented, device=device)
        for p, (b, bbox) in zip(pre, binaries):
            p['binary'] = b
        if all_feats_compute or ect_features:
            ects = ect_torch([b for b, bbox in binaries], sf._regular_directions(36, dims=2), T=32, device=device)
            for p, e in zip(pre, ects):
                p['ect'] = e

    per_obj = [[None] * len(frames) for _ in range(len(boundaries))]
    for (b, jj, tt), c, p in zip(tqdm(samples), contours, pre):
        per_obj[b][jj] = sf._morphology_features_sample_fast(c, imshape, opts, pre=p)
    return sf._assemble_fast_features(per_obj, len(boundaries), len(frames))


def compute_boundary_texture_features_torch(vid, boundaries,
                                            use_gradient_vid=False,
                                            compute_all_feats =True,
                                            compute_intensity_feats=True,
                                            compute_contour_intensity_feats=True,
                                            n_contours=3,
                                            n_angles=1,
                                            angle_start=None,
                                            compute_sift=True,
                                            siftshape=(64,64),
                                            compute_haralick=True,
                                            haralick_distance=15,
                                            timesample=None,
                                            use_mask=True,
                                            masked_sift_resize='cap',
                                            masks=None,
                                            masked_haralick_levels=8,
                                            masked_haralick_range=None,
                                            device=None):
    """ PyTorch variant of SPOT.Features.features.compute_boundary_texture_features with the same parameters and outputs.

    The contour masks and regional windows (identical rasterisation), the Haralick features and the (unmasked) SIFT
    descriptors are batched over all samples on the device. The masked SIFT descriptors, the distance-halving fallback of
    the masked Haralick features and the intensity statistics are computed on the CPU as in the original.

    Parameters
    ----------
    (see compute_boundary_texture_features)
    device : None or str or torch.device
        the torch device. None uses cuda if available, else the CPU.

    Returns
    -------
    out_array : (n_organoids, n_frames, n_features) array
    out_feat_labels : (n_features,) str array
    out_feat_labels_norm : (n_features,) bool array
    """
    import skimage.transform as sktform
    from tqdm import tqdm
    # C order: the features are sensitive at the last bit to the memory layout of the contours (e.g. reductions such as
    # the contour mean), which can flip ties on pixel-grid contours; a fixed layout makes all paths identical.
    boundaries = np.ascontiguousarray(boundaries)
    if timesample is None:
        timesample = 1
    vid_range = None
    if use_mask and masked_haralick_levels is not None:
        if masked_haralick_range is not None:
            vid_range = masked_haralick_range
        elif np.issubdtype(np.asarray(vid).dtype, np.integer):
            vid_range = (np.iinfo(np.asarray(vid).dtype).min, np.iinfo(np.asarray(vid).dtype).max)
        else:
            vid_range = (float(np.nanmin(vid)), float(np.nanmax(vid)))
    frames = np.arange(0, boundaries.shape[1], timesample)
    opts = dict(use_gradient_vid=use_gradient_vid, compute_all_feats=compute_all_feats,
                compute_intensity_feats=compute_intensity_feats,
                compute_contour_intensity_feats=compute_contour_intensity_feats, n_contours=n_contours,
                n_angles=n_angles, angle_start=angle_start, compute_sift=compute_sift, siftshape=siftshape,
                compute_haralick=compute_haralick, haralick_distance=haralick_distance, use_mask=use_mask,
                masked_sift_resize=masked_sift_resize, masked_haralick_levels=masked_haralick_levels,
                vid_range=vid_range)
    samples = _valid_samples(boundaries, frames)
    contours = [boundaries[b][tt] for b, jj, tt in samples]
    fg = lambda tt: (np.asarray(masks[tt]) > 0) if (use_mask and masks is not None) else None

    pre = [dict(binary_bbox=bb) for bb in contour_to_binary_torch(contours, device=device)]
    if compute_all_feats or compute_contour_intensity_feats:
        for p, tile in zip(pre, radial_windows_torch(vid.shape[1:], n_contours, n_angles, contours,
                                                     zero_angle=angle_start, device=device)):
            p['tile'] = tile

    do_haralick = compute_all_feats or compute_haralick
    do_sift = (compute_all_feats or compute_sift) and not use_mask
    har_ims, har_d, har_k, sift_ims, sift_k = [], [], [], [], []
    for k, ((b, jj, tt), c) in enumerate(zip(samples, contours)):
        crop = sf._texture_crop(vid[tt], fg(tt), c, opts, binary_bbox=pre[k]['binary_bbox'])
        if crop is None:
            continue
        vid_cnt, cnt_binary, bbox, max_dist = crop
        if do_haralick:
            im, d = sf._texture_haralick_input(vid_cnt, cnt_binary, max_dist, opts)
            if use_mask:
                # as _masked_haralick: grey levels shifted to >= 1, outside the mask 0 and ignored
                im = im.astype(np.int32) + 1
                im[cnt_binary==0] = 0
            har_ims.append(im); har_d.append(int(d)); har_k.append(k)
        if do_sift:
            sift_ims.append(sktform.resize(vid_cnt, siftshape, preserve_range=True)); sift_k.append(k)

    if len(har_ims) > 0:
        feats, empty = haralick_torch(har_ims, har_d, ignore_zeros=use_mask, device=device)
        for k, f, e in zip(har_k, feats, empty):
            if use_mask:
                if e:
                    continue  # empty direction: the CPU _masked_haralick halves the distance
                f = f.copy(); f[5] = f[5] - 2  # undo the +1 grey level shift in sum_average, as _masked_haralick
            pre[k]['haralick'] = f
    if len(sift_ims) > 0:
        for k, f in zip(sift_k, sift_torch(np.stack(sift_ims), device=device)):
            pre[k]['sift'] = f

    siftextractor = sf.SingleSiftExtractor(siftshape[0])
    per_obj = [[None] * len(frames) for _ in range(len(boundaries))]
    for (b, jj, tt), c, p in zip(tqdm(samples), contours, pre):
        per_obj[b][jj] = sf._texture_features_sample_fast(vid[tt], fg(tt), c, opts, siftextractor, pre=p)
    return sf._assemble_fast_features(per_obj, len(boundaries), len(frames))


def compute_boundary_motion_features_torch(vid_flow,
                                           boundaries,
                                           compute_all_feats =True,
                                           compute_global_feats=True,
                                           compute_contour_feats=True,
                                           n_contour_feat_bins=8,
                                           cnt_sigma=3.,
                                           n_contours=3,
                                           n_angles=1,
                                           angle_start=None,
                                           pixel_res=1.,
                                           time_res=1.,
                                           compute_sift_feats=True,
                                           siftshape=(64,64),
                                           timesample=None,
                                           use_mask=True,
                                           masked_sift_resize='cap',
                                           masks=None,
                                           device=None,
                                           shear_features=True,
                                           directional_motion_sift=True,
                                           directional_sift_sigma=1.):
    """ PyTorch variant of SPOT.Features.features.compute_boundary_motion_features with the same parameters and outputs.

    Curl and divergence (once per frame), the contour masks and regional windows (identical rasterisation) and the
    (unmasked) SIFT descriptors are computed on the device, batched over all samples. The masked SIFT descriptors, speed
    statistics and density estimates are computed on the CPU as in the original.

    Parameters
    ----------
    (see compute_boundary_motion_features)
    device : None or str or torch.device
        the torch device. None uses cuda if available, else the CPU.

    Returns
    -------
    out_array : (n_organoids, n_frames-1, n_features) array
    out_feat_labels : (n_features,) str array
    out_feat_labels_norm : (n_features,) bool array
    """
    from tqdm import tqdm
    # C order: the features are sensitive at the last bit to the memory layout of the contours (e.g. reductions such as
    # the contour mean), which can flip ties on pixel-grid contours; a fixed layout makes all paths identical.
    boundaries = np.ascontiguousarray(boundaries)
    if timesample is None:
        timesample = 1
    frames = np.arange(0, boundaries.shape[1]-1, timesample)
    opts = dict(compute_all_feats=compute_all_feats, compute_global_feats=compute_global_feats,
                compute_contour_feats=compute_contour_feats, n_contour_feat_bins=n_contour_feat_bins,
                cnt_sigma=cnt_sigma, n_contours=n_contours, n_angles=n_angles, angle_start=angle_start,
                pixel_res=pixel_res, time_res=time_res, compute_sift_feats=compute_sift_feats, siftshape=siftshape,
                use_mask=use_mask, masked_sift_resize=masked_sift_resize, shear_features=shear_features,
                directional_motion_sift=directional_motion_sift, directional_sift_sigma=directional_sift_sigma)
    samples = _valid_samples(boundaries, frames, need_next=True)
    contours = [boundaries[b][tt] for b, jj, tt in samples]
    fg = lambda tt: (np.asarray(masks[tt]) > 0) if (use_mask and masks is not None) else None
    by_frame = {}
    for k, (b, jj, tt) in enumerate(samples):
        by_frame.setdefault(tt, []).append(k)

    pre = [dict(binary_bbox=bb) for bb in contour_to_binary_torch(contours, device=device)]
    if compute_all_feats or compute_contour_feats:
        for p, tile in zip(pre, radial_windows_torch(vid_flow.shape[1:-1], n_contours, n_angles, contours,
                                                     zero_angle=angle_start, device=device)):
            p['tile'] = tile

    def frame_fields(tt):
        flow_frame = np.asarray(vid_flow[tt]).copy()
        curl, div = curl_div_torch(flow_frame, time_res, device=device)
        # the shear rate on the CPU, as the other implementations (identical results)
        shear = (sf._shear_vector_flow(flow_frame) / float(time_res)) if shear_features else None
        return flow_frame, curl, div, shear

    siftextractor = sf.SingleSiftExtractor(siftshape[0])
    # pass 1: the images of the unmasked SIFT descriptors, batched on the device
    if (compute_all_feats or compute_sift_feats) and not use_mask and not directional_motion_sift:
        sift_ims, sift_k = [], []
        for tt in sorted(by_frame):
            flow_frame, curl, div, shear = frame_fields(tt)
            for k in by_frame[tt]:
                b = samples[k][0]
                n_before = len(sift_ims)
                sf._motion_features_sample_fast(flow_frame, curl, div, fg(tt), boundaries[b][tt], boundaries[b][tt+1],
                                                opts, siftextractor, pre=pre[k], sift_inputs=sift_ims,
                                                shear_vid_frame=shear)
                if len(sift_ims) > n_before:
                    sift_k.append(k)
        if len(sift_ims) > 0:
            sifts = sift_torch(np.stack(sift_ims), device=device).reshape(len(sift_k), 4 if shear_features else 3, -1)
            for k, s3 in zip(sift_k, sifts):
                pre[k]['sift'] = list(s3)

    # pass 2: all features
    per_obj = [[None] * len(frames) for _ in range(len(boundaries))]
    for tt in tqdm(sorted(by_frame)):
        flow_frame, curl, div, shear = frame_fields(tt)
        for k in by_frame[tt]:
            b, jj, _ = samples[k]
            per_obj[b][jj] = sf._motion_features_sample_fast(flow_frame, curl, div, fg(tt), boundaries[b][tt],
                                                             boundaries[b][tt+1], opts, siftextractor, pre=pre[k],
                                                             shear_vid_frame=shear)
    return sf._assemble_fast_features(per_obj, len(boundaries), len(frames))
