# Copyright 2026 Linum Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File: t5.py
Description: Class that encapsulates the T5 model and tokenizer with text truncation.
"""
from typing import List, Optional
import torch

from transformers import AutoTokenizer, T5EncoderModel, logging as transformers_logging


class LinumT5EncoderModel:

    def __init__(
            self,
            checkpoint_path: str,
            tokenizer_path: str,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
            max_length: int = 256,
    ):
        """
        Initialize the T5 encoder model for text embedding.

        Args:
            checkpoint_path (str):
                Path to the pre-trained T5 encoder model checkpoint.
            tokenizer_path (str):
                Path to the T5 tokenizer.
            dtype (torch.dtype):
                Data type for model weights. Default is torch.bfloat16.
            device (torch.device):
                Device to load the model on. Default is current CUDA device.
            max_length (int):
                Maximum token sequence length. Default is 256.
        """
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length

        # Suppress transformers warnings during loading
        original_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()

        # Load Model
        self.model = T5EncoderModel.from_pretrained(checkpoint_path).eval().requires_grad_(False)
        self.model.to(self.device, dtype=self.dtype)

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Restore original verbosity
        transformers_logging.set_verbosity(original_verbosity)

    def __call__(self,
                 text: List[str],
                 device: Optional[torch.device] = None) -> List[torch.Tensor]:
        """
        Encode text prompts into T5 embeddings.

        Args:
            text (List[str]):
                List of text prompts to encode.
            device (Optional[torch.device]):
                Device for output tensors. Defaults to the model's device.

        Returns:
            List[torch.Tensor]:
                List of embedding tensors, one per input text. Each tensor has
                shape (seq_len, hidden_dim), trimmed to the actual token length
                (excluding padding).
        """
        device = self.device if device is None else device

        # Tokenize the list of texts with max_length
        tokenized = self.tokenizer(text, return_tensors='pt', padding=True,
                                  truncation=True, max_length=self.max_length)
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        # Get the context with attention mask
        context = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Extract sequence lengths from attention mask
        seq_lens = attention_mask.sum(dim=1).long()

        # Return a list of tensors, each trimmed to its actual sequence length
        return [context[i, :seq_len] for i, seq_len in enumerate(seq_lens)]
