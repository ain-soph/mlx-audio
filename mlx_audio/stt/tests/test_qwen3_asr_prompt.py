import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel


def _prompt_text(system_prompt, language=None, num_audio_tokens=2):
    model = Qwen3ASRModel.__new__(Qwen3ASRModel)
    model.config = SimpleNamespace(support_languages=["English", "Chinese"])
    model._tokenizer = Mock()
    model._tokenizer.encode = Mock(return_value=np.zeros((1, 1), dtype=np.int32))
    model._build_prompt(num_audio_tokens, language, system_prompt)
    return model._tokenizer.encode.call_args.args[0]


class TestQwen3ASRPrompt(unittest.TestCase):
    """The prompt must match the official Qwen3-ASR chat_template.json rendering."""

    AUDIO = "<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>"

    def test_context_has_no_trailing_newline(self):
        self.assertEqual(
            _prompt_text("交易 停滞"),
            "<|im_start|>system\n交易 停滞<|im_end|>\n"
            f"<|im_start|>user\n{self.AUDIO}<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_empty_context(self):
        self.assertEqual(
            _prompt_text(None),
            "<|im_start|>system\n<|im_end|>\n"
            f"<|im_start|>user\n{self.AUDIO}<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_forced_language_prefix(self):
        self.assertTrue(
            _prompt_text("terms", language="chinese").endswith(
                "<|im_start|>assistant\nlanguage Chinese<asr_text>"
            )
        )


if __name__ == "__main__":
    unittest.main()
