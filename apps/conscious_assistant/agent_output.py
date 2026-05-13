import re
from typing import List
from typing import Sequence
from typing import Tuple


_AGENT_OUTPUT_BLOCK_PATTERN = re.compile(
    r"(?m)^(thought|say):[ \t]*(.*?)(?=^\s*(?:thought|say):|\Z)",
    re.S,
)


def parse_agent_output_blocks(text: str) -> Tuple[List[str], List[str]]:
    """
    Split a normalized agent response into thought blocks and speech blocks.

    The top agent is expected to emit lines that begin with `thought:` or
    `say:`. This helper extracts those payloads without pulling in the heavier
    Flask runtime.
    """
    thoughts: List[str] = []
    speeches: List[str] = []

    for kind, raw in _AGENT_OUTPUT_BLOCK_PATTERN.findall(text or ""):
        content = raw.lstrip()
        if not content:
            continue
        if kind == "thought":
            thoughts.append(content)
        else:
            speeches.append(content)

    return thoughts, speeches


def combine_speech_blocks(speeches: Sequence[str]) -> Tuple[str, str]:
    """
    Combine multiple `say:` blocks into UI text and a single spoken utterance.

    The UI can preserve the multi-line structure, while TTS benefits from a
    single request so the robot does not pause between adjacent `say:` blocks.
    """
    cleaned_blocks = [str(block).strip() for block in speeches if str(block).strip()]
    if not cleaned_blocks:
        return "", ""

    display_text = "\n".join(cleaned_blocks)
    spoken_text = " ".join(
        block.replace("\n", " ").strip()
        for block in cleaned_blocks
    )
    spoken_text = re.sub(r"\s+", " ", spoken_text).strip()
    return display_text, spoken_text
