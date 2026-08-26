import os
from pathlib import Path
from PIL import Image

def convert_image_to_pgm(image_path: Path, output_path: Path = None, format_type: str = "P5") -> Path:
    """
    Converts a single image to grayscale .pgm format.
    
    :param image_path: Path to the source image file
    :param output_path: Path to save the .pgm file (defaults to same name in same directory or output dir)
    :param format_type: 'P5' (binary raw format) or 'P2' (plain text ASCII format)
    :return: Path to the generated .pgm file
    """
    image_path = Path(image_path)
    if output_path is None:
        out_file = image_path.with_suffix(".pgm")
    elif output_path.is_dir() or output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"{image_path.stem}.pgm"
    else:
        out_file = output_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {image_path.name} -> {out_file.name} (Grayscale {format_type} PGM)...")

    with Image.open(image_path) as img:
        # Convert image to 8-bit grayscale ('L' mode)
        gray_img = img.convert("L")
        width, height = gray_img.size

        if format_type == "P5":
            # Default PIL .save with .pgm produces binary P5 format
            gray_img.save(out_file)
        elif format_type == "P2":
            # Plain ASCII P2 format with fast chunked buffering
            raw_bytes = gray_img.tobytes()
            stride = 16 * 8000  # Process 128,000 pixels per chunk
            with open(out_file, "w", encoding="ascii") as f:
                f.write(f"P2\n# Created by convert_to_pgm.py\n{width} {height}\n255\n")
                for start in range(0, len(raw_bytes), stride):
                    sub = raw_bytes[start:start + stride]
                    lines = []
                    for i in range(0, len(sub), 16):
                        chunk = sub[i:i + 16]
                        lines.append(" ".join(str(b) for b in chunk))
                    f.write("\n".join(lines) + "\n")
        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Use 'P5' or 'P2'.")

    print(f" Saved: {out_file} ({out_file.stat().st_size:,} bytes)")
    return out_file

def convert_images_to_pgm(image_dir: Path, output_dir: Path = None, format_type: str = "P5"):
    """
    Converts all images in `image_dir` to grayscale .pgm format.
    
    :param image_dir: Path to directory containing images
    :param output_dir: Path to save .pgm files (defaults to image_dir)
    :param format_type: 'P5' (binary raw format, compact) or 'P2' (plain text ASCII format)
    """
    image_dir = Path(image_dir)
    if output_dir is None:
        output_dir = image_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for file_path in image_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            convert_image_to_pgm(file_path, output_dir / f"{file_path.stem}.pgm", format_type=format_type)

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    img_dir = base_dir / "images"
    
    # Convert image_2.jpg to P2 ASCII format
    images = ["image_2.jpg"]
    for image in images:
        image_path = img_dir / image
        if image_path.exists():
            convert_image_to_pgm(image_path, format_type="P2")
        else:
            print(f"File not found: {image_path}")
