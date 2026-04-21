"""OpticalAgent: renders code-heavy tool outputs as images.

Inherits DefaultAgent, overrides execute_actions to post-process
observation outputs. Code-heavy outputs (code_ratio > 0.2 and
len > 2000) are rendered as images.

Key constraint: OpenAI tool messages only support string content,
not multimodal. So we keep the tool message as a placeholder text
and inject a follow-up user message containing the image.
"""

import base64
import io
from pathlib import Path

from minisweagent.agents.default import DefaultAgent

from render_code_to_image import render_text_to_images


CODE_STARTERS = ('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'return ',
                 'while ', 'with ', 'try:', 'except ', 'elif ', 'else:', '#', '    ')

RENDER_MIN_CHARS = 2000
RENDER_MIN_CODE_RATIO = 0.2


def is_code_heavy(text: str) -> bool:
    lines = text.split('\n')
    if len(lines) < 5:
        return False
    code_lines = sum(1 for l in lines if l.strip().startswith(CODE_STARTERS))
    return code_lines / len(lines) > RENDER_MIN_CODE_RATIO


def should_render(text: str) -> bool:
    return len(text) > RENDER_MIN_CHARS and is_code_heavy(text)


def render_to_base64_list(text: str, header: str = "") -> list[str]:
    images = render_text_to_images(
        text, font_size=12, page_width_chars=120, lines_per_image=80, header=header,
    )
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result


class OpticalAgent(DefaultAgent):
    """Agent that renders code-heavy tool outputs as images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_stats = {
            "observations_total": 0,
            "observations_rendered": 0,
            "observations_text": 0,
            "total_images": 0,
        }

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions, render code-heavy outputs as images via user messages."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = [self.env.execute(action) for action in actions]

        # Identify which outputs need rendering BEFORE formatting
        render_flags = []
        image_data = []
        for output in outputs:
            self.render_stats["observations_total"] += 1
            raw = output.get("output", "")

            if should_render(raw):
                n_lines = len(raw.split('\n'))
                header = f"Command output ({n_lines} lines)"
                b64_list = render_to_base64_list(raw, header=header)
                render_flags.append(True)
                image_data.append(b64_list)
                # Replace the output with a placeholder
                output["output"] = f"[Code output ({n_lines} lines) rendered as image in the next message]"
                self.render_stats["observations_rendered"] += 1
                self.render_stats["total_images"] += len(b64_list)
            else:
                render_flags.append(False)
                image_data.append(None)
                self.render_stats["observations_text"] += 1

        # Format observation messages normally (tool messages with text)
        obs_messages = self.model.format_observation_messages(
            message, outputs, self.get_template_vars()
        )
        self.add_messages(*obs_messages)

        # For each rendered output, inject a user message with the images
        for flag, b64_list in zip(render_flags, image_data):
            if flag and b64_list:
                content = [{"type": "text", "text": "The code output from the previous command is shown in the image(s) below:"}]
                for b64 in b64_list:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                user_img_msg = {"role": "user", "content": content}
                self.add_messages(user_img_msg)

        return obs_messages  # Return the observation messages (agent loop expects this)
