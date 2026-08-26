from pathlib import Path
from PIL import Image

def convert_pgm_to_jpg(pgm_path: Path, output_path: Path = None, quality: int = 95) -> Path:
    """
    Converts a single .pgm image (P5 Binary or P2 ASCII) to .jpg format.
    
    :param pgm_path: Path to the source .pgm file
    :param output_path: Destination path for the .jpg file (defaults to same name with .jpg)
    :param quality: JPEG compression quality (1-100, default 95)
    :return: Path to the generated .jpg file
    """
    pgm_path = Path(pgm_path)
    if output_path is None:
        out_file = pgm_path.with_suffix(".jpg")
    elif output_path.is_dir():
        out_file = output_path / f"{pgm_path.stem}.jpg"
    else:
        out_file = output_path

    print(f"Converting {pgm_path.name} -> {out_file.name} (JPEG)...")
    with Image.open(pgm_path) as img:
        # Convert Grayscale ('L') to RGB for standard JPEG compatibility
        rgb_img = img.convert("RGB")
        rgb_img.save(out_file, "JPEG", quality=quality)
    
    print(f" Saved: {out_file} ({out_file.stat().st_size:,} bytes, {img.size[0]}x{img.size[1]})")
    return out_file

def convert_all_pgm_to_jpg(image_dir: Path, output_dir: Path = None, quality: int = 95):
    """
    Converts all .pgm images in `image_dir` to .jpg format.
    """
    image_dir = Path(image_dir)
    if output_dir is None:
        output_dir = image_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pgm_files = sorted(image_dir.glob("*.pgm"))
    
    if not pgm_files:
        print(f"No .pgm files found in {image_dir}")
        return

    print(f"Found {len(pgm_files)} .pgm file(s) in {image_dir}. Converting to JPEG...")
    for pgm_file in pgm_files:
        convert_pgm_to_jpg(pgm_file, output_dir / f"{pgm_file.stem}.jpg", quality=quality)

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    img_dir = base_dir / "images"

    # Targets: newly generated processed ROI, brightened, and investigation demo PGM images
    targets = [
        "image_1_roi.pgm",
        "image_2_roi.pgm",
        "image_2_brightened.pgm",
        "demo_flat_gray.pgm",
        "demo_checkerboard.pgm",
        "demo_gradient.pgm",
        "demo_concentric_circles.pgm",
    ]

    print("=" * 70)
    print("           PGM TO JPG CONVERTER (PIL / PILLOW)           ")
    print("=" * 70)

    for target in targets:
        target_path = img_dir / target
        if target_path.exists():
            convert_pgm_to_jpg(target_path)
        else:
            print(f"Skipping {target}: file not found in {img_dir}")
