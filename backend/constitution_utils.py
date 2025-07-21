import os
from functools import lru_cache
import re

CONSTITUTION_DIR = os.path.join(os.path.dirname(__file__), 'constitutions')

@lru_cache(maxsize=8)
def get_constitution(platform: str) -> str:
    """
    Load and cache the constitution text for the given platform (e.g., 'linkedin', 'twitter').
    Raises FileNotFoundError if the constitution file does not exist.
    """
    filename = f"{platform.lower()}.txt"
    path = os.path.join(CONSTITUTION_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Constitution file not found for platform: {platform} (expected at {path})")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_constitution_defaults(platform: str) -> dict:
    """
    Extracts key defaults (tone, structure, audience, etc.) from the constitution text for the given platform.
    Returns a dictionary of defaults if found, else empty dict.
    """
    text = get_constitution(platform)
    defaults = {}
    # Extract Tone
    tone_match = re.search(r"Voice & Tone:\s*([\s\S]+?)(?:\n\n|\n\d+\.|\n\-|$)", text)
    if tone_match:
        defaults['tone'] = tone_match.group(1).strip()
    # Extract Structure/Framework
    structure_match = re.search(r"Framework for Every Post[\s\S]+?Section\s+Description([\s\S]+?)(?:\n\n|\n\d+\.|\n\-|$)", text)
    if structure_match:
        defaults['structure'] = structure_match.group(1).strip()
    # Extract Audience (if present)
    audience_match = re.search(r"target audience:?\s*([\w,\s]+)", text, re.IGNORECASE)
    if audience_match:
        defaults['audience'] = audience_match.group(1).strip()
    return defaults 