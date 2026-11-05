"""Encoder-token counter; mirrors the caption templating and cropping in
pan/models/text_encoders/__init__.py. Local tokenizer assets only."""
import threading
from typing import List
VIDEO_TEMPLATE_SYS = "You are a helpful assistant. Generate the video with the following instructions."

class EncoderTokenCounter:
    """Tokens the video backbone actually receives for a caption string."""

    def __init__(self, tokenizer_dir: str):
        from transformers import AutoTokenizer  # heavy; only when enabled
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        probe = self._ids("x")
        marker = self.tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        self.crop = next(i + len(marker) for i in range(len(probe))
                         if probe[i:i + len(marker)] == marker)
        self.lock = threading.Lock()

    def _ids(self, text: str) -> List[int]:
        ids = self.tok.apply_chat_template(
            [{"role": "system", "content": VIDEO_TEMPLATE_SYS}, {"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True)
        return ids["input_ids"] if hasattr(ids, "keys") else ids

    def __call__(self, text: str) -> int:
        with self.lock:
            return len(self._ids(text)) - self.crop

