from pathlib import Path
from PIL import Image

def convert_to_ppm(image_path: Path, output_path: Path, format_type: str = "P6") -> Path:
    """
    Converts an image to PPM format (P6 Binary 24-bit RGB or P3 ASCII).
    
    :param image_path: Path to source image (e.g. .jpg, .png, .pgm)
    :param output_path: Destination path for .ppm file
    :param format_type: 'P6' (raw binary RGB) or 'P3' (plain ASCII RGB)
    :return: Path to saved .ppm file
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {image_path} -> {output_path} (PPM {format_type})...")
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size

        if format_type == "P6":
            # Pillow natively saves P6 binary PPM
            rgb_img.save(output_path)
        elif format_type == "P3":
            # Plain ASCII P3 PPM
            raw_bytes = rgb_img.tobytes()
            stride = 15 * 6000  # Process chunks of pixels
            with open(output_path, "w", encoding="ascii") as f:
                f.write(f"P3\n# Created by convert_to_ppm.py\n{width} {height}\n255\n")
                for start in range(0, len(raw_bytes), stride):
                    sub = raw_bytes[start:start + stride]
                    lines = []
                    for i in range(0, len(sub), 15):
                        chunk = sub[i:i + 15]
                        lines.append(" ".join(str(b) for b in chunk))
                    f.write("\n".join(lines) + "\n")
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    print(f" Saved: {output_path} ({output_path.stat().st_size:,} bytes, {width}x{height})")
    return output_path

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    
    # Locate source images from lab_1/images
    src_dirs = [base_dir / "lab_1" / "images", base_dir / "images"]
    dest_dirs = [base_dir / "lab_2" / "images", base_dir / "lab_1" / "images"]

    print("=" * 70)
    print("           PPM (PORTABLE PIXMAP) IMAGE CONVERTER (P6/P3)         ")
    print("=" * 70)

    # Find image_1.jpg or image_1.pgm in source directories
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        
        for name in ["image_1.jpg", "image_2.jpg"]:
            src_file = src_dir / name
            if src_file.exists():
                for dest_dir in dest_dirs:
                    dest_file = dest_dir / f"{src_file.stem}.ppm"
                    convert_to_ppm(src_file, dest_file, format_type="P6")
