#!/usr/bin/env python3
import argparse, glob, sys
import numpy as np
import cv2
import os

def parse_args():
    ap = argparse.ArgumentParser(description="Stereo calibration with bad-pair rejection.")
    ap.add_argument("--pattern", nargs=2, required=True,
                    help='Two globs: --pattern "left_*.png" "right_*.png"')
    ap.add_argument("--board", nargs=2, type=int, required=True,
                    help="Checkerboard inner corners: COLS ROWS (e.g., 9 7)")
    ap.add_argument("--square", type=float, required=True,
                    help="Checkerboard square size in meters (e.g., 0.015)")
    ap.add_argument("--mono-thresh", type=float, default=1.0,
                    help="Per-view mono RMS threshold (px) to keep a frame")
    ap.add_argument("--stereo-thresh", type=float, default=2.0,
                    help="Target stereo RMS (px) to stop dropping pairs")
    ap.add_argument("--max-iters", type=int, default=3,
                    help="Max iterations of dropping worst pairs")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="stereoRectify alpha: 0=crop, 1=full, -1=auto (default 1)")
    ap.add_argument("--free-intrinsics", action="store_true",
                    help="Let stereoCalibrate refine K,D (recommended if baseline is tiny)")
    ap.add_argument("--out", type=str, default="stereo_rectify_maps.yml",
                    help="Output YAML (OpenCV FileStorage format)")
    return ap.parse_args()

def make_object_points(cols, rows, square):
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square)
    return objp

def find_corners(paths, pattern_size):
    corners_list, size = [], None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read {p}, skipping.")
            corners_list.append(None)
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        # Prefer SB detector if available, else fallback
        if hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(img, pattern_size)
        else:
            ok, corners = cv2.findChessboardCorners(
                img, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        if not ok:
            corners_list.append(None)
            continue
        cv2.cornerSubPix(img, corners, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        corners_list.append(corners.astype(np.float32))
    return corners_list, size

def calibrate_mono(obj_all, img_all, size):
    valid = [i for i,(o,ip) in enumerate(zip(obj_all, img_all)) if o is not None and ip is not None]
    if not valid: return None
    obj = [obj_all[i] for i in valid]
    img = [img_all[i] for i in valid]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, size, None, None, flags=cv2.CALIB_RATIONAL_MODEL
    )
    per_view = []
    for o, ip, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        e = proj.reshape(-1,2) - ip.reshape(-1,2)
        per_view.append(np.sqrt(np.mean(np.sum(e*e, axis=1))))
    return {"valid_idx": valid, "rms": float(rms), "K": K, "D": D, "per_view": np.array(per_view)}

def filter_by_mono(left_cal, right_cal, mono_thresh, total_count):
    keep_L = {left_cal["valid_idx"][i] for i,e in enumerate(left_cal["per_view"]) if e <= mono_thresh}
    keep_R = {right_cal["valid_idx"][i] for i,e in enumerate(right_cal["per_view"]) if e <= mono_thresh}
    kept = sorted(keep_L.intersection(keep_R))
    print(f"Left kept {len(keep_L)}/{len(left_cal['valid_idx'])}; "
          f"Right kept {len(keep_R)}/{len(right_cal['valid_idx'])}; "
          f"Intersect kept {len(kept)}/{total_count} (thr={mono_thresh})")
    return kept

def gather(obj_all, img_all, idxs):
    return [obj_all[i] for i in idxs], [img_all[i] for i in idxs]

def average_mono_pair_error(left_cal, right_cal, kept_idxs):
    errL = {i:e for i,e in zip(left_cal["valid_idx"], left_cal["per_view"])}
    errR = {i:e for i,e in zip(right_cal["valid_idx"], right_cal["per_view"])}
    scores = []
    for i in kept_idxs:
        eL, eR = errL.get(i, np.inf), errR.get(i, np.inf)
        scores.append((i, 0.5*(eL+eR)))
    scores.sort(key=lambda x: x[1], reverse=True)  # worst first
    return scores

def save_yaml(path, K1,D1,K2,D2,R,T,R1,R2,P1,P2,Q,m1l,m2l,m1r,m2r):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for k,v in [("K1",K1),("D1",D1),("K2",K2),("D2",D2),("R",R),("T",T),
                ("R1",R1),("R2",R2),("P1",P1),("P2",P2),("Q",Q),
                ("map1_l",m1l),("map2_l",m2l),("map1_r",m1r),("map2_r",m2r)]:
        fs.write(k, v)
    fs.release()

def main():
    args = parse_args()
    pattL, pattR = args.pattern
    cols, rows = args.board
    square = args.square

    left_paths  = sorted(glob.glob(pattL))
    right_paths = sorted(glob.glob(pattR))
    if not left_paths or not right_paths:
        print("No images found for left or right pattern."); sys.exit(1)
    if len(left_paths) != len(right_paths):
        n = min(len(left_paths), len(right_paths))
        print(f"[WARN] Count mismatch L={len(left_paths)} R={len(right_paths)}; truncating to {n}")
        left_paths, right_paths = left_paths[:n], right_paths[:n]

    pattern_size = (cols, rows)
    objp = make_object_points(cols, rows, square)

    L_pts, sizeL = find_corners(left_paths, pattern_size)
    R_pts, sizeR = find_corners(right_paths, pattern_size)
    if sizeL is None or sizeR is None:
        print("Could not detect any corners."); sys.exit(1)
    if sizeL != sizeR:
        print(f"[ERROR] Different image sizes L{sizeL} vs R{sizeR}. Aborting."); sys.exit(1)
    img_size = sizeL

    N = len(left_paths)
    obj_all = [objp for _ in range(N)]
    L_all, R_all = L_pts, R_pts

    calL = calibrate_mono(obj_all, L_all, img_size)
    calR = calibrate_mono(obj_all, R_all, img_size)
    if calL is None or calR is None:
        print("Insufficient detections for mono calibration."); sys.exit(1)
    print(f"Left mono RMS: {calL['rms']:.3f}")
    print(f"Right mono RMS: {calR['rms']:.3f}")

    kept = filter_by_mono(calL, calR, args.mono_thresh, N)
    if len(kept) < 5:
        print("Too few views after mono filtering."); sys.exit(1)

    # Intrinsics from the kept set
    obj_kept, L_kept = gather(obj_all, L_all, kept)
    _, K1, D1, _, _ = cv2.calibrateCamera(obj_kept, L_kept, img_size, None, None)
    obj_kept, R_kept = gather(obj_all, R_all, kept)
    _, K2, D2, _, _ = cv2.calibrateCamera(obj_kept, R_kept, img_size, None, None)

    # Prepare loop state
    curr_kept = kept.copy()
    best = None

    # Choose stereo flags
    if args.free_intrinsics:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL
    else:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                 cv2.CALIB_FIX_FOCAL_LENGTH |
                 cv2.CALIB_FIX_PRINCIPAL_POINT |
                 cv2.CALIB_RATIONAL_MODEL)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)

    for it in range(args.max_iters + 1):
        obj_kept, L_kept = gather(obj_all, L_all, curr_kept)
        _, R_kept = gather(obj_all, R_all, curr_kept)

        # Start from mono intrinsics each iter
        K1g, D1g = K1.copy(), D1.copy()
        K2g, D2g = K2.copy(), D2.copy()

        rms, K1s, D1s, K2s, D2s, R, T, E, F = cv2.stereoCalibrate(
            obj_kept, L_kept, R_kept,
            K1g, D1g, K2g, D2g,
            img_size, flags=flags, criteria=criteria
        )
        print(f"[iter {it}] Stereo RMS: {rms:.3f} with {len(curr_kept)} pairs")

        if (best is None) or (rms < best["rms"]):
            best = {"rms": rms, "K1": K1s, "D1": D1s, "K2": K2s, "D2": D2s,
                    "R": R, "T": T, "kept": curr_kept.copy()}

        if rms <= args.sterio_thresh if hasattr(args, 'sterio_thresh') else args.stereo_thresh or it == args.max_iters:
            break

        # Drop worst by avg mono error (computed from original mono fits)
        scores = average_mono_pair_error(calL, calR, curr_kept)
        worst_idx = scores[0][0]
        print(f"  Dropping worst pair idx={worst_idx} (avg mono err={scores[0][1]:.3f})")
        curr_kept.remove(worst_idx)
        if len(curr_kept) < 5:
            print("Too few pairs remain; stopping."); break

    # Rectify from the best solution
    K1b, D1b = best["K1"], best["D1"]
    K2b, D2b = best["K2"], best["D2"]
    Rb, Tb = best["R"],  best["T"]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1b, D1b, K2b, D2b, img_size, Rb, Tb, alpha=args.alpha
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1b, D1b, R1, P1, img_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2b, D2b, R2, P2, img_size, cv2.CV_16SC2)

    save_yaml(args.out, K1b, D1b, K2b, D2b, Rb, Tb, R1, R2, P1, P2, Q, map1_l, map2_l, map1_r, map2_r)
    print(f"✅ Saved {args.out}")
    print(f"Kept {len(best['kept'])} pairs; final stereo RMS={best['rms']:.3f}")
    print(f"Final baseline: {np.linalg.norm(Tb):.3f} m")

if __name__ == "__main__":
    main()
