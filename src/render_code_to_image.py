"""Render code/text to monospace PNG images for optical compression experiments."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def find_monospace_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a monospace font on the system."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Fallback: use PIL default (not ideal but works)
    print("WARNING: No system monospace font found, using default")
    return ImageFont.load_default()


def wrap_line(line: str, max_chars: int) -> list[str]:
    """Wrap a single line to max_chars width."""
    if len(line) <= max_chars:
        return [line]
    wrapped = []
    while len(line) > max_chars:
        wrapped.append(line[:max_chars])
        line = line[max_chars:]
    if line:
        wrapped.append(line)
    return wrapped


def render_text_to_images(
    text: str,
    *,
    font_size: int = 14,
    page_width_chars: int = 100,
    lines_per_image: int = 80,
    dpi: int = 150,
    header: str = "",
    padding: int = 10,
) -> list[Image.Image]:
    """Render text to one or more PNG images.

    Returns a list of PIL Image objects.
    """
    font = find_monospace_font(font_size)

    # Measure character dimensions
    bbox = font.getbbox("M")
    char_w = bbox[2] - bbox[0]
    line_h = int((bbox[3] - bbox[1]) * 1.4)  # 1.4x line spacing

    img_w = padding * 2 + char_w * page_width_chars
    header_h = line_h + 4 if header else 0

    # Wrap lines
    raw_lines = text.split("\n")
    wrapped_lines = []
    for line in raw_lines:
        # Replace tabs with spaces
        line = line.replace("\t", "    ")
        wrapped_lines.extend(wrap_line(line, page_width_chars))

    # Split into chunks
    chunks = []
    for i in range(0, len(wrapped_lines), lines_per_image):
        chunks.append(wrapped_lines[i : i + lines_per_image])

    images = []
    for chunk_idx, chunk in enumerate(chunks):
        img_h = padding * 2 + header_h + line_h * len(chunk)
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)

        y = padding
        if header:
            h_text = header if len(chunks) == 1 else f"{header} (part {chunk_idx + 1}/{len(chunks)})"
            draw.rectangle([0, 0, img_w, header_h + padding], fill="#e0e0e0")
            draw.text((padding, y), h_text, fill="black", font=font)
            y += header_h

        for line in chunk:
            draw.text((padding, y), line, fill="black", font=font)
            y += line_h

        # Set DPI metadata
        images.append(img)

    return images


def render_and_save(
    text: str,
    output_dir: Path,
    prefix: str,
    **kwargs,
) -> list[Path]:
    """Render text and save images to disk. Returns list of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = render_text_to_images(text, **kwargs)
    paths = []
    for i, img in enumerate(images):
        suffix = f"_p{i}" if len(images) > 1 else ""
        path = output_dir / f"{prefix}{suffix}.png"
        img.save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    # Quick test
    sample = '''def fibonacci(n):
    """Return the n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Test
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
'''
    out = Path(__file__).parent / "images" / "test"
    paths = render_and_save(sample, out, "test", header="test_fibonacci.py", font_size=14)
    for p in paths:
        print(f"Saved: {p} ({p.stat().st_size / 1024:.1f} KB)")
