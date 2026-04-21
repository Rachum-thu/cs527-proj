"""Preprocessor that converts code blocks in problem statements to images.

This implements Plan A: instance-level replacement for mini-swe-agent.
Code blocks in the issue description are rendered as images and embedded
using MSWEA_MULTIMODAL_CONTENT tags, which the existing multimodal
infrastructure in litellm_model.py expands into OpenAI image_url format.
"""

import base64
import io
import re
from pathlib import Path

from render_code_to_image import render_text_to_images


# Regex to find markdown code blocks: ```lang\n...\n```
CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def render_code_to_base64(
    code: str,
    header: str = "",
    font_size: int = 12,
    page_width_chars: int = 120,
    lines_per_image: int = 80,
) -> list[str]:
    """Render code text to base64-encoded PNG images."""
    images = render_text_to_images(
        code,
        font_size=font_size,
        page_width_chars=page_width_chars,
        lines_per_image=lines_per_image,
        header=header,
    )
    b64_list = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_list.append(base64.b64encode(buf.getvalue()).decode())
    return b64_list


def make_multimodal_tag(b64_data: str) -> str:
    """Wrap base64 image data in MSWEA_MULTIMODAL_CONTENT tag."""
    return (
        f"<MSWEA_MULTIMODAL_CONTENT>"
        f"<CONTENT_TYPE>image_url</CONTENT_TYPE>"
        f"data:image/png;base64,{b64_data}"
        f"</MSWEA_MULTIMODAL_CONTENT>"
    )


def preprocess_optical_file(problem_statement: str, **render_kwargs) -> str:
    """Optical-File: render each code block as a separate image.

    Each code block becomes one (or more) images, preserving the
    semantic boundary of each code snippet.
    """
    def replace_code_block(match):
        lang = match.group(1) or "code"
        code = match.group(2)
        header = f"Code block ({lang})"

        b64_images = render_code_to_base64(code, header=header, **render_kwargs)
        tags = "\n".join(make_multimodal_tag(b64) for b64 in b64_images)
        return f"[The following code is shown as an image]\n{tags}"

    return CODE_BLOCK_RE.sub(replace_code_block, problem_statement)


def preprocess_optical_fixed(problem_statement: str, **render_kwargs) -> str:
    """Optical-Fixed: concatenate all code blocks, then split by fixed line count.

    All code blocks are merged into one stream, then split into
    equal-sized 80-line segments regardless of original code block boundaries.
    """
    # Extract all code blocks
    code_blocks = []
    for match in CODE_BLOCK_RE.finditer(problem_statement):
        lang = match.group(1) or "code"
        code = match.group(2)
        code_blocks.append((lang, code))

    if not code_blocks:
        return problem_statement  # No code blocks, return as-is

    # Concatenate all code into one stream
    combined = "\n\n".join(f"# --- {lang} block ---\n{code}" for lang, code in code_blocks)

    # Render as fixed-length images
    b64_images = render_code_to_base64(combined, header="Combined code context", **render_kwargs)
    tags = "\n".join(make_multimodal_tag(b64) for b64 in b64_images)

    # Remove all code blocks from text, append images at the end
    text_only = CODE_BLOCK_RE.sub("[code block rendered as image below]", problem_statement)
    return f"{text_only}\n\n[All code blocks are shown as images below]\n{tags}"


def preprocess_text(problem_statement: str, **render_kwargs) -> str:
    """Text baseline: no preprocessing, return as-is."""
    return problem_statement


STRATEGIES = {
    "text": preprocess_text,
    "file": preprocess_optical_file,
    "fixed": preprocess_optical_fixed,
}


if __name__ == "__main__":
    # Quick test with a sample problem statement
    sample = """Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True, False],
       [False, False, False,  True]])
```

This feels like a bug to me, but I might be missing something?
"""
    # Test Optical-File
    result_file = preprocess_optical_file(sample)
    n_tags = result_file.count("MSWEA_MULTIMODAL_CONTENT")
    print(f"Optical-File: {n_tags // 2} image tags injected")
    print(f"Text length: {len(sample)} -> {len(result_file)} chars")

    # Test Optical-Fixed
    result_fixed = preprocess_optical_fixed(sample)
    n_tags_fixed = result_fixed.count("MSWEA_MULTIMODAL_CONTENT")
    print(f"Optical-Fixed: {n_tags_fixed // 2} image tags injected")

    # Verify tags are well-formed
    import re
    from minisweagent.models.utils.openai_multimodal import DEFAULT_MULTIMODAL_REGEX
    matches = list(re.finditer(DEFAULT_MULTIMODAL_REGEX, result_file))
    print(f"Multimodal regex matches: {len(matches)} (should equal image count)")
    for i, m in enumerate(matches):
        print(f"  Match {i}: type={m.group(1)}, data_len={len(m.group(2))}")
