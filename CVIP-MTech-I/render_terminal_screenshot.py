from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/consolab.ttf"

def render_terminal_image(
    text_content: str,
    output_path: Path,
    title: str = "Terminal Output",
    font_size: int = 15,
    padding: int = 24,
    scale: int = 2
) -> Path:
    """
    Renders terminal text into a high-DPI macOS/Linux-style terminal window screenshot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = text_content.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    
    # Load fonts
    try:
        font = ImageFont.truetype(FONT_PATH, font_size * scale)
        font_bold = ImageFont.truetype(FONT_BOLD_PATH, font_size * scale)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    # Calculate line dimensions
    sample_bbox = font.getbbox("A")
    char_width = sample_bbox[2] - sample_bbox[0]
    line_height = int((sample_bbox[3] - sample_bbox[1]) * 1.45)
    
    max_line_len = max(len(line) for line in lines) if lines else 80
    if max_line_len < 70:
        max_line_len = 70

    header_height = 36 * scale
    content_width = (max_line_len * char_width) + (padding * 2 * scale)
    content_height = (len(lines) * line_height) + (padding * 2 * scale) + header_height

    # Theme colors
    bg_color = (26, 27, 38)       # TokyoNight dark navy
    header_color = (36, 40, 59)   # Slightly lighter header
    text_color = (192, 202, 245)  # Light lavender/white
    border_color = (65, 72, 104)  # Muted border
    title_color = (122, 162, 247) # Accent blue title

    img = Image.new("RGB", (content_width, content_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw header bar
    draw.rectangle([(0, 0), (content_width, header_height)], fill=header_color)
    draw.line([(0, header_height), (content_width, header_height)], fill=border_color, width=scale)

    # Window traffic light buttons
    btn_radius = 6 * scale
    btn_y = header_height // 2
    buttons = [
        (18 * scale, (255, 95, 86)),   # Red close
        (34 * scale, (255, 189, 46)),  # Yellow minimize
        (50 * scale, (39, 201, 63)),   # Green maximize
    ]
    for btn_x, col in buttons:
        draw.ellipse(
            [(btn_x - btn_radius, btn_y - btn_radius), (btn_x + btn_radius, btn_y + btn_radius)],
            fill=col
        )

    # Draw title centered in header
    title_font = ImageFont.truetype(FONT_BOLD_PATH, 13 * scale) if FONT_BOLD_PATH else font
    title_bbox = title_font.getbbox(title)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (content_width - title_w) // 2
    draw.text((title_x, (header_height - (title_bbox[3] - title_bbox[1])) // 2), title, fill=title_color, font=title_font)

    # Draw terminal text content
    curr_y = header_height + (padding * scale)
    x_offset = padding * scale

    for line in lines:
        col = text_color
        f = font

        # Syntax color cues
        stripped = line.strip()
        if stripped.startswith("===") or stripped.startswith("###"):
            col = (122, 162, 247)  # Blue banner
            f = font_bold
        elif "[SUCCESS]" in line or "[VERIFIED" in line or "[Exported" in line:
            col = (158, 206, 106)  # Green success
            f = font_bold
        elif "[ERROR]" in line or "[FATAL]" in line or "[FAILED" in line:
            col = (247, 118, 142)  # Red error
            f = font_bold
        elif "[Transforming]" in line or "[Generating]" in line or "[Exporting]" in line or "[Analytical" in line or "[Mathematical" in line:
            col = (224, 175, 104)  # Orange info
            f = font_bold
        elif stripped.startswith("*") or stripped.startswith("-"):
            col = (187, 154, 247)  # Light purple for list bullets
        elif "#" in line and "|" in line:
            col = (115, 218, 202)  # Cyan histogram bar

        draw.text((x_offset, curr_y), line, fill=col, font=f)
        curr_y += line_height

    # Outer border
    draw.rectangle([(0, 0), (content_width - 1, content_height - 1)], outline=border_color, width=scale)

    img.save(output_path, "PNG", optimize=True)
    print(f"  [Generated Terminal Screenshot] {output_path.name} ({img.size[0]}x{img.size[1]})")
    return output_path

def split_and_render_lab_outputs():
    base_dir = Path(__file__).parent
    
    # Read lab_1 stdout
    lab_1_text = (base_dir / "lab_1_stdout.txt").read_text(encoding="utf-8")
    lab_2_text = (base_dir / "lab_2_stdout.txt").read_text(encoding="utf-8")

    screenshots_dir = base_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    lab1_overleaf_images = base_dir / "overleaf" / "lab_1" / "images"
    lab2_overleaf_images = base_dir / "overleaf" / "lab_2" / "images"

    lab1_overleaf_images.mkdir(parents=True, exist_ok=True)
    lab2_overleaf_images.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("       GENERATING TERMINAL OUTPUT SCREENSHOTS (HIGH-DPI PNG)     ")
    print("=" * 70)

    # 1. Full Screenshots
    print("\nRendering Full Lab 1 & Lab 2 Output Screenshots...")
    lab1_full = render_terminal_image(lab_1_text, screenshots_dir / "lab_1_output.png", "CVIP Lab 1 - Complete Grayscale Analysis & ROI Engine")
    lab2_full = render_terminal_image(lab_2_text, screenshots_dir / "lab_2_output.png", "CVIP Lab 2 - Netpbm Header Parser & Size Verifier")

    # 2. Sectional Screenshots for Lab 1
    print("\nRendering Lab 1 Sectional Screenshots...")
    p1_text = "\n".join(lab_1_text.split("################################################################################\n        PART B:")[0].splitlines())
    p2_text = "################################################################################\n        PART B:" + lab_1_text.split("################################################################################\n        PART B:")[1].split("################################################################################\n        PART C:")[0]
    p3_text = "################################################################################\n        PART C:" + lab_1_text.split("################################################################################\n        PART C:")[1].split("################################################################################\n  PART E:")[0]
    p4_text = "################################################################################\n  PART E:" + lab_1_text.split("################################################################################\n  PART E:")[1]

    img_l1_p1 = render_terminal_image(p1_text, screenshots_dir / "lab_1_output_part1.png", "Lab 1 - Global Grayscale Analysis & Comparison")
    img_l1_p2 = render_terminal_image(p2_text, screenshots_dir / "lab_1_output_part2.png", "Lab 1 - Localized ROI Selection & Evaluation")
    img_l1_p3 = render_terminal_image(p3_text, screenshots_dir / "lab_1_output_part3.png", "Lab 1 - Brightening Transformation & Histogram Shift")
    img_l1_p4 = render_terminal_image(p4_text, screenshots_dir / "lab_1_output_part4.png", "Lab 1 - Equal Mean Intensity Investigation")

    # 3. Sectional Screenshots for Lab 2
    print("\nRendering Lab 2 Sectional Screenshots...")
    l2_sec1 = lab_2_text.split("################################################################################\n        SECTION 2:")[0]
    l2_sec2 = "################################################################################\n        SECTION 2:" + lab_2_text.split("################################################################################\n        SECTION 2:")[1].split("################################################################################\n        INVESTIGATION:")[0]
    l2_sec3 = "################################################################################\n        INVESTIGATION:" + lab_2_text.split("################################################################################\n        INVESTIGATION:")[1].split("################################################################################\n                    CRITICAL-THINKING")[0]
    l2_sec4 = "################################################################################\n                    CRITICAL-THINKING" + lab_2_text.split("################################################################################\n                    CRITICAL-THINKING")[1]

    img_l2_p1 = render_terminal_image(l2_sec1, screenshots_dir / "lab_2_output_part1.png", "Lab 2 - PGM Grayscale Analysis & Size Verification")
    img_l2_p2 = render_terminal_image(l2_sec2, screenshots_dir / "lab_2_output_part2.png", "Lab 2 - PPM Truecolor RGB Analysis & Verification")
    img_l2_p3 = render_terminal_image(l2_sec3, screenshots_dir / "lab_2_output_part3.png", "Lab 2 - Extension vs Header Format Investigation")
    img_l2_p4 = render_terminal_image(l2_sec4, screenshots_dir / "lab_2_output_part4.png", "Lab 2 - Critical-Thinking 640x480 Size Calculations")

    # Copy all generated output screenshots to overleaf directories
    for png in [lab1_full, img_l1_p1, img_l1_p2, img_l1_p3, img_l1_p4]:
        shutil.copy(png, lab1_overleaf_images / png.name)

    for png in [lab2_full, img_l2_p1, img_l2_p2, img_l2_p3, img_l2_p4]:
        shutil.copy(png, lab2_overleaf_images / png.name)

    print("\n[SUCCESS] All output screenshots generated in root directory and overleaf folders.")

if __name__ == "__main__":
    split_and_render_lab_outputs()
