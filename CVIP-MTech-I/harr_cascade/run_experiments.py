"""
run_experiments.py: Systematic Face Detection Benchmarking Suite

Executes automated experiments across:
1. Three Distinct Parameter Configurations:
   - Config 1: Baseline (scaleFactor=1.10, minNeighbors=4)
   - Config 2: High Sensitivity (scaleFactor=1.05, minNeighbors=2)
   - Config 3: High Precision (scaleFactor=1.30, minNeighbors=8)
   - Config 4: Fine-Scale Strict (scaleFactor=1.03, minNeighbors=6)
2. Three Controlled Physical Perturbation Axes:
   - Scale Variation (0.15x, 0.25x, 0.50x, 1.00x, 1.75x)
   - Illumination Variation (Severe Underexposure, Moderate Dim, Normal, Overexposure, Specular Burn, Directional Shadow)
   - Orientation Variation (0 deg, +/-15 deg, +/-30 deg, +45 deg, +90 deg, 180 deg, Semi-Profile, Full Profile)
3. Evaluates Ground Truth, IoU, TP, FP, FN, and Inference Latency (ms).
4. Generates publication-ready comparative visualization grids and summary CSV/LaTeX tables.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configure matplotlib for headless rendering
plt.switch_backend('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

OUTPUT_DIR = "harr_cascade/outputs"
OVERLEAF_IMG_DIR = "overleaf/harr_cascade/images"
DATASET_DIR = "harr_cascade/dataset"
CASCADE_XML = "harr_cascade/Face_Detection/haarcascade_frontalface_default.xml"
PROFILE_XML = cv2.data.haarcascades + "haarcascade_profileface.xml" if hasattr(cv2, 'data') else None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLEAF_IMG_DIR, exist_ok=True)

# Define the Parameter Configurations
CONFIGS = {
    "Config 1 (Baseline)": {
        "scaleFactor": 1.10,
        "minNeighbors": 4,
        "minSize": (30, 30),
        "desc": "sF=1.10, mN=4 (Standard Balanced)"
    },
    "Config 2 (High Sensitivity)": {
        "scaleFactor": 1.05,
        "minNeighbors": 2,
        "minSize": (20, 20),
        "desc": "sF=1.05, mN=2 (Permissive / High Recall)"
    },
    "Config 3 (High Precision)": {
        "scaleFactor": 1.30,
        "minNeighbors": 8,
        "minSize": (40, 40),
        "desc": "sF=1.30, mN=8 (Conservative / High Precision)"
    },
    "Config 4 (Fine Strict)": {
        "scaleFactor": 1.03,
        "minNeighbors": 6,
        "minSize": (30, 30),
        "desc": "sF=1.03, mN=6 (Fine-grain Strict)"
    }
}


def load_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_XML)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from {CASCADE_XML}")
    return cascade


def compute_iou(boxA, boxB):
    """Compute Intersection-over-Union between two boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


def evaluate_detections(detected_boxes, gt_boxes, iou_thresh=0.35):
    """
    Match detected boxes with ground truth boxes.
    Returns: tp, fp, fn, max_iou, status
    """
    if len(gt_boxes) == 0:
        # No ground truth face
        fp = len(detected_boxes)
        status = "SUCCESS (No Face)" if fp == 0 else f"FAILED ({fp} False Positives)"
        return 0, fp, 0, 0.0, status

    matched_gt = set()
    tp = 0
    fp = 0
    max_iou = 0.0

    for d_box in detected_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            iou = compute_iou(d_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_iou > max_iou:
            max_iou = best_iou
        if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    if tp == len(gt_boxes) and fp == 0:
        status = "SUCCESS (Exact Match)"
    elif tp > 0 and fp > 0:
        status = f"PARTIAL ({tp} TP, {fp} FP)"
    elif tp > 0 and fn > 0:
        status = f"PARTIAL ({tp}/{len(gt_boxes)} detected)"
    else:
        status = "FAILED (Missed Detection)"

    return tp, fp, fn, max_iou, status


def draw_detection_box(img, box, color=(0, 255, 0), label="Face"):
    """Draw rectangle with label tag on image copy."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    tag_y = max(y - 4, th + 2)
    cv2.rectangle(img, (x, tag_y - th - 2), (x + tw + 2, tag_y + 2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, label, (x + 1, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# -------------------------------------------------------------------------------------------------
# EXPERIMENT 1: SCALE VARIATION
# -------------------------------------------------------------------------------------------------
def run_scale_experiment(base_img, base_gt, cascade):
    """
    Evaluate face detection across multiple scale factors.
    base_img: Lena portrait (512x512)
    base_gt: [219, 203, 169, 169]
    """
    scales = [0.18, 0.30, 0.50, 1.00, 1.60]
    scale_labels = [
        "Tiny (0.18x, ~30px face)",
        "Small (0.30x, ~50px face)",
        "Medium (0.50x, ~85px face)",
        "Standard (1.00x, ~169px face)",
        "Large (1.60x, ~270px face)"
    ]

    results = []
    annotated_grid = {cfg_name: [] for cfg_name in CONFIGS.keys()}

    for s_idx, (scale, label) in enumerate(zip(scales, scale_labels)):
        # Resize image
        new_w = int(base_img.shape[1] * scale)
        new_h = int(base_img.shape[0] * scale)
        scaled_img = cv2.resize(base_img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        
        # Scale ground truth box
        scaled_gt = [int(v * scale) for v in base_gt]
        
        for cfg_name, cfg in CONFIGS.items():
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=cfg["scaleFactor"],
                minNeighbors=cfg["minNeighbors"],
                minSize=cfg["minSize"]
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            faces_list = faces.tolist() if hasattr(faces, 'tolist') else list(faces)
            tp, fp, fn, max_iou, status = evaluate_detections(faces_list, [scaled_gt])

            # Prepare visualization
            disp_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB).copy()
            # Draw GT in dashed / cyan
            draw_detection_box(disp_img, scaled_gt, color=(0, 255, 255), label="GT")
            # Draw detections
            for f_box in faces_list:
                iou = compute_iou(f_box, scaled_gt)
                color = (0, 255, 0) if iou >= 0.35 else (255, 0, 0)
                draw_detection_box(disp_img, f_box, color=color, label=f"Det (IoU:{iou:.2f})")

            annotated_grid[cfg_name].append({
                'img': disp_img,
                'scale': scale,
                'label': label,
                'faces': len(faces_list),
                'latency': elapsed,
                'status': status
            })

            results.append({
                'Experiment': 'Scale Variation',
                'Variation': label,
                'Scale': scale,
                'Face_Width_px': scaled_gt[2],
                'Config': cfg_name,
                'scaleFactor': cfg["scaleFactor"],
                'minNeighbors': cfg["minNeighbors"],
                'Faces_Detected': len(faces_list),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Max_IoU': round(max_iou, 3),
                'Latency_ms': round(elapsed, 2),
                'Status': status
            })

    return results, annotated_grid, scale_labels


# -------------------------------------------------------------------------------------------------
# EXPERIMENT 2: ILLUMINATION VARIATION
# -------------------------------------------------------------------------------------------------
def adjust_illumination(img, delta_brightness, gamma):
    """Apply linear brightness offset and non-linear power-law gamma correction."""
    # Convert to float [0, 1]
    norm = img.astype(np.float32) / 255.0
    # Apply gamma
    gamma_corr = np.power(norm, gamma)
    # Apply brightness shift
    shifted = gamma_corr * 255.0 + delta_brightness
    # Clip to valid 8-bit range
    adjusted = np.clip(shifted, 0, 255).astype(np.uint8)
    return adjusted


def apply_directional_shadow(img):
    """Simulate strong directional side spotlight with cast shadow gradient."""
    h, w = img.shape[:2]
    # Gradient from 1.3 (left highlight) down to 0.15 (right deep shadow)
    gradient = np.linspace(1.3, 0.15, w).reshape(1, w, 1)
    gradient = np.repeat(gradient, h, axis=0)
    shadowed = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    return shadowed


def run_illumination_experiment(base_img, base_gt, cascade):
    """
    Evaluate face detection across illumination degradations.
    """
    variations = [
        ("Severe Underexposure", lambda img: adjust_illumination(img, -70, 2.5)),
        ("Moderate Dim Light", lambda img: adjust_illumination(img, -35, 1.5)),
        ("Normal Diffuse Lighting", lambda img: img.copy()),
        ("Moderate Overexposure", lambda img: adjust_illumination(img, +40, 0.6)),
        ("Specular Burn / Flash", lambda img: adjust_illumination(img, +80, 0.35)),
        ("Directional Side Shadow", apply_directional_shadow)
    ]

    results = []
    annotated_grid = {cfg_name: [] for cfg_name in CONFIGS.keys()}
    var_labels = [name for name, _ in variations]

    for label, transform in variations:
        mod_img = transform(base_img)
        gray = cv2.cvtColor(mod_img, cv2.COLOR_BGR2GRAY)

        for cfg_name, cfg in CONFIGS.items():
            t0 = time.perf_counter()
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=cfg["scaleFactor"],
                minNeighbors=cfg["minNeighbors"],
                minSize=cfg["minSize"]
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            faces_list = faces.tolist() if hasattr(faces, 'tolist') else list(faces)
            tp, fp, fn, max_iou, status = evaluate_detections(faces_list, [base_gt])

            disp_img = cv2.cvtColor(mod_img, cv2.COLOR_BGR2RGB).copy()
            draw_detection_box(disp_img, base_gt, color=(0, 255, 255), label="GT")
            for f_box in faces_list:
                iou = compute_iou(f_box, base_gt)
                color = (0, 255, 0) if iou >= 0.35 else (255, 0, 0)
                draw_detection_box(disp_img, f_box, color=color, label=f"Det (IoU:{iou:.2f})")

            annotated_grid[cfg_name].append({
                'img': disp_img,
                'label': label,
                'faces': len(faces_list),
                'latency': elapsed,
                'status': status
            })

            results.append({
                'Experiment': 'Illumination Variation',
                'Variation': label,
                'Scale': 1.0,
                'Face_Width_px': base_gt[2],
                'Config': cfg_name,
                'scaleFactor': cfg["scaleFactor"],
                'minNeighbors': cfg["minNeighbors"],
                'Faces_Detected': len(faces_list),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Max_IoU': round(max_iou, 3),
                'Latency_ms': round(elapsed, 2),
                'Status': status
            })

    return results, annotated_grid, var_labels


# -------------------------------------------------------------------------------------------------
# EXPERIMENT 3: ORIENTATION VARIATION
# -------------------------------------------------------------------------------------------------
def rotate_image_and_box(img, box, angle_deg):
    """Rotate image and compute bounding rectangle around rotated ground truth face."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Calculate new bounding dimensions to avoid clipping
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    
    # Adjust rotation matrix with translation
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Transform ground truth 4 corners
    bx, by, bw, bh = box
    corners = np.array([
        [bx, by, 1.0],
        [bx + bw, by, 1.0],
        [bx + bw, by + bh, 1.0],
        [bx, by + bh, 1.0]
    ]).T
    transformed_corners = np.dot(M, corners)
    min_x = int(np.min(transformed_corners[0, :]))
    max_x = int(np.max(transformed_corners[0, :]))
    min_y = int(np.min(transformed_corners[1, :]))
    max_y = int(np.max(transformed_corners[1, :]))
    
    rotated_gt = [min_x, min_y, max_x - min_x, max_y - min_y]
    return rotated_img, rotated_gt


def run_orientation_experiment(base_img, base_gt, cascade):
    """
    Evaluate face detection across in-plane rotations and out-of-plane poses.
    """
    rotations = [
        ("0 deg (Frontal Upright)", 0),
        ("+15 deg (Slight Tilt)", 15),
        ("-15 deg (Opposite Tilt)", -15),
        ("+30 deg (Moderate Tilt)", 30),
        ("+45 deg (Steep Tilt)", 45),
        ("+90 deg (Sideways)", 90),
        ("180 deg (Inverted)", 180)
    ]

    results = []
    annotated_grid = {cfg_name: [] for cfg_name in CONFIGS.keys()}
    rot_labels = [name for name, _ in rotations]

    for label, angle in rotations:
        rot_img, rot_gt = rotate_image_and_box(base_img, base_gt, angle)
        gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)

        for cfg_name, cfg in CONFIGS.items():
            t0 = time.perf_counter()
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=cfg["scaleFactor"],
                minNeighbors=cfg["minNeighbors"],
                minSize=cfg["minSize"]
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            faces_list = faces.tolist() if hasattr(faces, 'tolist') else list(faces)
            tp, fp, fn, max_iou, status = evaluate_detections(faces_list, [rot_gt])

            disp_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB).copy()
            draw_detection_box(disp_img, rot_gt, color=(0, 255, 255), label="GT")
            for f_box in faces_list:
                iou = compute_iou(f_box, rot_gt)
                color = (0, 255, 0) if iou >= 0.35 else (255, 0, 0)
                draw_detection_box(disp_img, f_box, color=color, label=f"Det (IoU:{iou:.2f})")

            annotated_grid[cfg_name].append({
                'img': disp_img,
                'label': label,
                'faces': len(faces_list),
                'latency': elapsed,
                'status': status
            })

            results.append({
                'Experiment': 'Orientation Variation',
                'Variation': label,
                'Scale': 1.0,
                'Face_Width_px': rot_gt[2],
                'Config': cfg_name,
                'scaleFactor': cfg["scaleFactor"],
                'minNeighbors': cfg["minNeighbors"],
                'Faces_Detected': len(faces_list),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Max_IoU': round(max_iou, 3),
                'Latency_ms': round(elapsed, 2),
                'Status': status
            })

    return results, annotated_grid, rot_labels


# -------------------------------------------------------------------------------------------------
# GRID VISUALIZATION RENDERERS
# -------------------------------------------------------------------------------------------------
def save_comparison_grid(annotated_grid, row_labels, title, filename):
    """Render a comprehensive N_variations x N_configs grid comparison image."""
    cfg_names = list(CONFIGS.keys())
    n_rows = len(row_labels)
    n_cols = len(cfg_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.2 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, label in enumerate(row_labels):
        for c_idx, cfg_name in enumerate(cfg_names):
            ax = axes[r_idx, c_idx]
            item = annotated_grid[cfg_name][r_idx]
            ax.imshow(item['img'])
            ax.set_xticks([])
            ax.set_yticks([])

            # Determine status badge color
            status = item['status']
            if "SUCCESS" in status:
                badge_color = "#107c41"  # Forest green
            elif "PARTIAL" in status:
                badge_color = "#d83b01"  # Orange-red
            else:
                badge_color = "#a80000"  # Crimson

            # Title on top row
            if r_idx == 0:
                ax.set_title(f"{cfg_name}\n({CONFIGS[cfg_name]['desc']})", fontsize=10, fontweight='bold', pad=8)
            
            # Left Y-label for row condition
            if c_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold', labelpad=6)

            # Subtitle at bottom of each cell
            info_txt = f"{status}\nFaces: {item['faces']} | {item['latency']:.1f}ms"
            ax.text(0.5, -0.12, info_txt, transform=ax.transAxes,
                    fontsize=8.5, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color, alpha=0.85, edgecolor='none'),
                    color='white', fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Copy to Overleaf images folder
    overleaf_path = os.path.join(OVERLEAF_IMG_DIR, filename)
    plt.figure()
    plt.close()
    import shutil
    shutil.copy2(out_path, overleaf_path)
    print(f"Saved visual grid: {out_path} and {overleaf_path}")


def plot_summary_charts(df):
    """Plot quantitative comparison charts for latency and detection success."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Chart 1: Average Latency per Configuration across all experiments
    avg_latency = df.groupby('Config')['Latency_ms'].mean().loc[list(CONFIGS.keys())]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(range(len(avg_latency)), avg_latency.values, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
    ax1.set_xticks(range(len(avg_latency)))
    ax1.set_xticklabels([c.replace(" (", "\n(") for c in avg_latency.index], fontsize=9.5, fontweight='bold')
    ax1.set_ylabel("Average Latency (ms)", fontsize=11, fontweight='bold')
    ax1.set_title("Detection Latency vs Parameter Configuration", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f} ms",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # Chart 2: Success / Partial / Failure Rates across configs
    def categorize_status(s):
        if "SUCCESS" in s:
            return "Success (TP>=1, FP=0)"
        elif "PARTIAL" in s:
            return "Partial (TP+FP)"
        else:
            return "Failed (Missed / FP only)"

    df['Outcome'] = df['Status'].apply(categorize_status)
    outcome_counts = pd.crosstab(df['Config'], df['Outcome']).loc[list(CONFIGS.keys())]

    outcome_colors = {'Success (TP>=1, FP=0)': '#28a745', 'Partial (TP+FP)': '#ffc107', 'Failed (Missed / FP only)': '#dc3545'}
    cols_present = [c for c in ['Success (TP>=1, FP=0)', 'Partial (TP+FP)', 'Failed (Missed / FP only)'] if c in outcome_counts.columns]
    
    outcome_counts[cols_present].plot(kind='bar', stacked=True, ax=ax2,
                                      color=[outcome_colors[c] for c in cols_present],
                                      edgecolor='black', linewidth=1.1, width=0.55)
    ax2.set_xticklabels([c.replace(" (", "\n(") for c in outcome_counts.index], rotation=0, fontsize=9.5, fontweight='bold')
    ax2.set_xlabel("")
    ax2.set_ylabel("Number of Test Scenarios", fontsize=11, fontweight='bold')
    ax2.set_title("Detection Outcome Distribution across All Tests (N=18)", fontsize=12, fontweight='bold')
    ax2.legend(title="Outcome", fontsize=9, loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_p = os.path.join(OUTPUT_DIR, "latency_and_accuracy_summary.png")
    plt.savefig(chart_p, dpi=200, bbox_inches='tight')
    plt.close()
    
    import shutil
    shutil.copy2(chart_p, os.path.join(OVERLEAF_IMG_DIR, "latency_and_accuracy_summary.png"))
    print(f"Saved summary charts: {chart_p}")


def main():
    print("=" * 80)
    print("      HAAR CASCADE FACE DETECTION EXPERIMENTAL BENCHMARK")
    print("      Evaluating Scale, Illumination, and Orientation Variations")
    print("=" * 80)

    cascade = load_cascade()
    print("Haar Cascade loaded successfully.")

    # Load baseline test portrait (Lena, 512x512)
    lena_path = os.path.join(DATASET_DIR, "frontal_portrait.jpg")
    base_img = cv2.imread(lena_path)
    if base_img is None:
        raise FileNotFoundError(f"Missing baseline image at {lena_path}")
    base_gt = [219, 203, 169, 169]  # [x, y, w, h] ground truth

    all_results = []

    # 1. Run Scale Experiment
    print("\n[1/3] Running Scale Variation Experiments...")
    res_scale, grid_scale, scale_labels = run_scale_experiment(base_img, base_gt, cascade)
    all_results.extend(res_scale)
    save_comparison_grid(grid_scale, scale_labels,
                         "Scale Variation Experiment: Multi-Scale Pyramid Sensitivity",
                         "scale_experiment_grid.png")

    # 2. Run Illumination Experiment
    print("\n[2/3] Running Illumination Variation Experiments...")
    res_illum, grid_illum, illum_labels = run_illumination_experiment(base_img, base_gt, cascade)
    all_results.extend(res_illum)
    save_comparison_grid(grid_illum, illum_labels,
                         "Illumination Variation Experiment: Exposure & Gradient Robustness",
                         "illumination_experiment_grid.png")

    # 3. Run Orientation Experiment
    print("\n[3/3] Running Orientation Variation Experiments...")
    res_orient, grid_orient, orient_labels = run_orientation_experiment(base_img, base_gt, cascade)
    all_results.extend(res_orient)
    save_comparison_grid(grid_orient, orient_labels,
                         "Orientation Variation Experiment: In-Plane Roll & Pose Rotation",
                         "orientation_experiment_grid.png")

    # Compile DataFrame and save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "experiments_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved quantitative results to {csv_path}")

    # Plot summary comparison charts
    plot_summary_charts(df)

    # Print summary text table
    print("\n" + "=" * 95)
    print(f"{'Experiment':<24} | {'Variation':<28} | {'Config':<26} | {'Status':<22} | {'IoU':<5} | {'ms':<6}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['Experiment']:<24} | {r['Variation']:<28} | {r['Config']:<26} | {r['Status']:<22} | {r['Max_IoU']:<5} | {r['Latency_ms']:<6}")
    print("=" * 95)

    print("\nBenchmark experiment completed successfully!")


if __name__ == "__main__":
    main()
