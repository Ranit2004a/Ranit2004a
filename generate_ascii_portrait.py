#!/usr/bin/env python3
"""
ASCII Portrait Generator for GitHub Profile (with Animation)
=============================================================
Generates production-quality, GitHub-profile-ready ASCII portrait PNGs
and animated GIFs from input portrait photographs using Pillow, NumPy, and OpenCV.

Features:
  - Deep-learning (YuNet) face detection with smart head/shoulder cropping
  - CLAHE local contrast enhancement & unsharp mask feature sharpening
  - Preserves hair texture, glasses, eyes, beard outline, and jawline
  - Character density mapping using only specified allowed characters:
    @%#*+=-:.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
  - Terminal-style window border framing
  - Static PNG outputs: assets/ascii-me.png & assets/ascii-me-green.png
  - Animated GIF outputs: assets/ascii-me.gif & assets/ascii-me-green.gif
  - README.md snippet generation
"""

import os
import sys
import random
import argparse
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Allowed character set strictly as specified in requirements
ALLOWED_CHARS = "@%#*+=-:.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Default theme settings
THEME_BLUE = {
    "png_filename": "ascii-me.png",
    "gif_filename": "ascii-me.gif",
    "title": "ascii-me",
    "bg": "#0d1117",
    "text": "#58a6ff",
    "scan_text": "#a5d6ff",
}

THEME_GREEN = {
    "png_filename": "ascii-me-green.png",
    "gif_filename": "ascii-me-green.gif",
    "title": "ascii-me-green",
    "bg": "black",
    "text": "#00ff41",
    "scan_text": "#66ff88",
}


def download_file(url: str, dest_path: str, description: str = "file") -> bool:
    """Download a file from URL if it does not already exist."""
    if os.path.exists(dest_path):
        return True
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"[*] Downloading {description} from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[+] Saved {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Warning: Could not download {description}: {e}")
        return False


def get_font_path(custom_font: str = None) -> str:
    """Find or download a high quality monospace font (JetBrains Mono / Consolas)."""
    if custom_font and os.path.exists(custom_font):
        return custom_font

    local_jetbrains = os.path.join("assets", "fonts", "JetBrainsMono-Regular.ttf")
    if os.path.exists(local_jetbrains):
        return local_jetbrains

    font_url = "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Regular.ttf"
    if download_file(font_url, local_jetbrains, "JetBrains Mono font"):
        return local_jetbrains

    system_fonts = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    for sf in system_fonts:
        if os.path.exists(sf):
            return sf

    raise FileNotFoundError("No suitable monospace font found on system.")


def compute_character_densities(font_path: str, font_size: int = 24) -> str:
    """Sort allowed character set strictly by rendered ink density (pixel coverage)."""
    font = ImageFont.truetype(font_path, font_size)
    densities = []

    for char in ALLOWED_CHARS:
        canvas = Image.new("L", (font_size, font_size), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((0, 0), char, fill=255, font=font)
        density = np.array(canvas).sum()
        densities.append((char, density))

    densities.sort(key=lambda item: item[1])
    return "".join([char for char, _ in densities])


def auto_detect_input_image() -> str:
    """Search common locations for a portrait photograph."""
    candidates = [
        "input.jpg",
        "input.jpeg",
        "input.png",
        "portrait.jpg",
        "portrait.png",
        "c:/Users/Asus/Downloads/LinkedinDP.jpeg",
        "c:/Users/Asus/OneDrive/Desktop/ID card.jpg",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    for f in os.listdir("."):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return f

    return None


def crop_shoulders_up(img: np.ndarray, model_path: str = "yunet.onnx") -> np.ndarray:
    """Detect subject's face using YuNet or contour fallback and crop from shoulders upward."""
    h, w = img.shape[:2]

    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    download_file(yunet_url, model_path, "YuNet Face Detector")

    face_detected = False
    fx, fy, fw, fh = 0, 0, 0, 0

    if os.path.exists(model_path):
        try:
            detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), score_threshold=0.6)
            _, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                face = faces[0]
                fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                face_detected = True
                print(f"[+] Face detected via YuNet: x={fx}, y={fy}, w={fw}, h={fh}")
        except Exception as e:
            print(f"[!] YuNet face detection notice: {e}")

    if not face_detected:
        print("[*] Face detector fallback: analyzing skin & upper body contours...")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            fx, fy, fw, fh = cv2.boundingRect(c)
            face_detected = True
            print(f"[+] Face contour detected: x={fx}, y={fy}, w={fw}, h={fh}")

    if face_detected:
        top = max(0, int(fy - 0.45 * fh))
        bottom = min(h, int(fy + fh + 1.35 * fh))

        crop_h = bottom - top
        target_crop_w = int(crop_h * 0.8)
        cx = int(fx + fw / 2)
        left = max(0, int(cx - target_crop_w / 2))
        right = min(w, int(cx + target_crop_w / 2))

        if right - left < target_crop_w:
            if left == 0:
                right = min(w, target_crop_w)
            elif right == w:
                left = max(0, w - target_crop_w)

        return img[top:bottom, left:right]

    print("[*] Using centered upper-body crop fallback.")
    return img[0:int(h * 0.75), int(w * 0.05):int(w * 0.95)]


def enhance_facial_features(bgr_img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale and enhance local contrast & sharpen facial details."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.5)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)

    sobelx = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    sobel_norm = np.uint8(np.clip(sobel / (sobel.max() + 1e-5) * 50, 0, 50))

    final = cv2.add(sharpened, sobel_norm)

    gamma = 1.15
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    final = cv2.LUT(final, table)

    return final


def image_to_ascii_grid(
    gray_img: np.ndarray,
    sorted_chars: str,
    target_width: int = 135,
    char_aspect: float = 0.55,
) -> list[str]:
    """Downsample image to ASCII resolution with aspect ratio correction."""
    h, w = gray_img.shape
    target_height = int((h / w) * target_width * char_aspect)

    resized = cv2.resize(gray_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    num_chars = len(sorted_chars)
    ascii_grid = []

    for r in range(target_height):
        row_chars = []
        for c in range(target_width):
            val = resized[r, c]
            idx = int((val / 255.0) * (num_chars - 1))
            row_chars.append(sorted_chars[idx])
        ascii_grid.append("".join(row_chars))

    return ascii_grid


def render_terminal_png(
    ascii_grid: list[str],
    output_path: str,
    bg_color: str,
    text_color: str,
    font_path: str,
    title: str = "ascii-me.png",
    font_size: int = 18,
):
    """Render ASCII character grid into a high-resolution static PNG image."""
    font = ImageFont.truetype(font_path, font_size)

    grid_w = len(ascii_grid[0])
    grid_h = len(ascii_grid)

    title_label = f"─[ {title} ]"
    top_bar = "┌" + title_label + "─" * (grid_w - len(title_label) + 2) + "┐"
    bottom_bar = "└" + "─" * (grid_w + 2) + "┘"

    framed_lines = [top_bar]
    for row in ascii_grid:
        framed_lines.append(f"│ {row} │")
    framed_lines.append(bottom_bar)

    bbox = font.getbbox("A")
    char_w = bbox[2] - bbox[0] + 1
    char_h = bbox[3] - bbox[1] + 4

    padding_x = 45
    padding_y = 45

    img_w = char_w * len(top_bar) + padding_x * 2
    img_h = char_h * len(framed_lines) + padding_y * 2

    canvas = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for i, line in enumerate(framed_lines):
        y = padding_y + i * char_h
        draw.text((padding_x, y), line, fill=text_color, font=font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, "PNG")
    print(f"[+] Output PNG saved: {output_path} ({img_w}x{img_h} px)")


def render_terminal_gif(
    ascii_grid: list[str],
    output_path: str,
    bg_color: str,
    text_color: str,
    scan_color: str,
    font_path: str,
    sorted_chars: str,
    title: str = "ascii-me.gif",
    font_size: int = 18,
    num_frames: int = 24,
    fps: int = 12,
):
    """
    Render ASCII character grid into an animated GIF with CRT scanline beam,
    subtle terminal character shimmer noise, and pulsing glows.
    """
    font = ImageFont.truetype(font_path, font_size)

    grid_w = len(ascii_grid[0])
    grid_h = len(ascii_grid)
    num_chars = len(sorted_chars)

    title_label = f"─[ {title} ]"
    top_bar = "┌" + title_label + "─" * (grid_w - len(title_label) + 2) + "┐"
    bottom_bar = "└" + "─" * (grid_w + 2) + "┘"

    bbox = font.getbbox("A")
    char_w = bbox[2] - bbox[0] + 1
    char_h = bbox[3] - bbox[1] + 4

    padding_x = 45
    padding_y = 45

    img_w = char_w * len(top_bar) + padding_x * 2
    img_h = char_h * (grid_h + 2) + padding_y * 2

    frames = []
    scan_step = (grid_h + 8) / num_frames

    for frame_idx in range(num_frames):
        scan_y = int(frame_idx * scan_step) - 4

        canvas = Image.new("RGB", (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(canvas)

        # Draw top border
        draw.text((padding_x, padding_y), top_bar, fill=text_color, font=font)

        for r in range(grid_h):
            row_chars = list(ascii_grid[r])

            # Apply subtle 2.5% random character glitch/shimmer noise
            for c in range(grid_w):
                if random.random() < 0.025:
                    c_idx = sorted_chars.find(row_chars[c])
                    if c_idx != -1:
                        shift = random.choice([-2, -1, 1, 2])
                        new_idx = max(0, min(num_chars - 1, c_idx + shift))
                        row_chars[c] = sorted_chars[new_idx]

            line_str = f"│ {''.join(row_chars)} │"
            y = padding_y + (r + 1) * char_h

            # Highlight scanline row beam
            if abs(r - scan_y) <= 1:
                draw.text((padding_x, y), line_str, fill=scan_color, font=font)
            else:
                draw.text((padding_x, y), line_str, fill=text_color, font=font)

        # Draw bottom border
        draw.text(
            (padding_x, padding_y + (grid_h + 1) * char_h), bottom_bar, fill=text_color, font=font
        )

        frames.append(canvas)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frame_duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )
    print(f"[+] Output Animated GIF saved: {output_path} ({num_frames} frames @ {fps} fps)")


def update_readme_file(include_gif: bool = True):
    """Create or update README.md with the center-aligned image snippet."""
    if include_gif:
        snippet = '<p align="center">\n  <img src="assets/ascii-me.gif" width="900" alt="Animated ASCII Portrait">\n</p>\n'
    else:
        snippet = '<p align="center">\n  <img src="assets/ascii-me.png" width="900" alt="ASCII Portrait">\n</p>\n'

    readme_path = "README.md"

    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        if 'src="assets/ascii-me' not in content:
            content = snippet + "\n" + content
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("[+] Updated README.md with ASCII portrait snippet.")
    else:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Profile ASCII Portrait\n\n" + snippet)
        print("[+] Created README.md with ASCII portrait snippet.")


def main():
    parser = argparse.ArgumentParser(description="GitHub Profile ASCII Portrait Generator")
    parser.add_argument("-i", "--input", help="Path to input portrait image file")
    parser.add_argument(
        "-w", "--width", type=int, default=135, help="ASCII character width (120-150, default 135)"
    )
    parser.add_argument("-f", "--font", help="Path to custom monospace font file")
    parser.add_argument(
        "-o", "--output-dir", default="assets", help="Directory to save output PNG & GIF files"
    )
    parser.add_argument(
        "--no-gif", action="store_true", help="Disable animated GIF generation"
    )
    parser.add_argument(
        "--frames", type=int, default=24, help="Number of animation frames (default 24)"
    )
    parser.add_argument(
        "--fps", type=int, default=12, help="Animation frames per second (default 12)"
    )

    args = parser.parse_args()

    target_width = max(120, min(150, args.width))
    input_path = args.input or auto_detect_input_image()
    if not input_path or not os.path.exists(input_path):
        print("[!] Error: No portrait photo found. Please specify --input path/to/photo.jpg")
        sys.exit(1)

    print(f"[*] Processing portrait photo: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"[!] Error: Unable to read image from {input_path}")
        sys.exit(1)

    cropped = crop_shoulders_up(img)
    enhanced = enhance_facial_features(cropped)

    font_path = get_font_path(args.font)
    print(f"[*] Using font: {font_path}")

    sorted_chars = compute_character_densities(font_path)
    print(f"[*] Character density map: {sorted_chars}")

    ascii_grid = image_to_ascii_grid(enhanced, sorted_chars, target_width=target_width)

    out_dir = args.output_dir
    path_blue_png = os.path.join(out_dir, THEME_BLUE["png_filename"])
    path_green_png = os.path.join(out_dir, THEME_GREEN["png_filename"])

    # Render PNGs
    render_terminal_png(
        ascii_grid,
        path_blue_png,
        bg_color=THEME_BLUE["bg"],
        text_color=THEME_BLUE["text"],
        font_path=font_path,
        title=THEME_BLUE["title"] + ".png",
    )

    render_terminal_png(
        ascii_grid,
        path_green_png,
        bg_color=THEME_GREEN["bg"],
        text_color=THEME_GREEN["text"],
        font_path=font_path,
        title=THEME_GREEN["title"] + ".png",
    )

    # Render GIFs (if enabled)
    if not args.no_gif:
        path_blue_gif = os.path.join(out_dir, THEME_BLUE["gif_filename"])
        path_green_gif = os.path.join(out_dir, THEME_GREEN["gif_filename"])

        render_terminal_gif(
            ascii_grid,
            path_blue_gif,
            bg_color=THEME_BLUE["bg"],
            text_color=THEME_BLUE["text"],
            scan_color=THEME_BLUE["scan_text"],
            font_path=font_path,
            sorted_chars=sorted_chars,
            title=THEME_BLUE["title"] + ".gif",
            num_frames=args.frames,
            fps=args.fps,
        )

        render_terminal_gif(
            ascii_grid,
            path_green_gif,
            bg_color=THEME_GREEN["bg"],
            text_color=THEME_GREEN["text"],
            scan_color=THEME_GREEN["scan_text"],
            font_path=font_path,
            sorted_chars=sorted_chars,
            title=THEME_GREEN["title"] + ".gif",
            num_frames=args.frames,
            fps=args.fps,
        )

    # Update README.md
    update_readme_file(include_gif=not args.no_gif)

    print("\n[+] ASCII Portrait generation complete!")


if __name__ == "__main__":
    main()
