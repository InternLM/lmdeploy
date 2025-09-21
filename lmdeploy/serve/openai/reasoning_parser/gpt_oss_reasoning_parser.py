# Reasoning parser for GPT-OSS style channels.
# Recognizes:
#   <|channel|>analysis<|message|> ... <|end|>
#   <|start|>assistant<|channel|>final<|message|> ...
#   <|channel|>final<|message|> ...
import re
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name='gpt-oss')
class GPTOssReasoningParser(ReasoningParser):
    ANALYSIS="<|channel|>analysis<|message|>"
    FINAL1="<|start|>assistant<|channel|>final<|message|>"
    FINAL2="<|channel|>final<|message|>"
    END="<|end|>"
    """Parser that splits LMDeploy-style channel tags into reasoning and final.

    Streaming behavior:
      - Tokens between analysis-start and <|end|> -> reasoning_content
      - Tokens after final-start markers -> content
    Non-streaming behavior:
      - Extract both channels from the full string with regex.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        if not self.model_tokenizer:
            raise ValueError('The model tokenizer must be passed to the ReasoningParser constructor.')

        # Raw tag strings
        self.analysis_start = '<|channel|>analysis<|message|>'
        self.final_start_with_assistant = '<|start|>assistant<|channel|>final<|message|>'
        self.final_start_plain = '<|channel|>final<|message|>'
        self.end_tag = '<|end|>'

        # For non-streaming extraction
        self._re_analysis = re.compile(r'<\|channel\|>analysis<\|message\|>(.*?)(?=(?:<\|end\|>|$))', re.S)
        # Allow both final markers
        self._re_final_with_assistant = re.compile(r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?=(?:<\|end\|>|$))', re.S)
        self._re_final_plain = re.compile(r'(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(.*?)(?=(?:<\|end\|>|$))', re.S)

        # Token ids for streaming checks
        self.analysis_start_id = self.vocab.get(self.analysis_start)
        self.final_start_with_assistant_id = self.vocab.get(self.final_start_with_assistant)
        self.final_start_plain_id = self.vocab.get(self.final_start_plain)
        self.end_id = self.vocab.get(self.end_tag)

    def _strip_tags(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        # Remove any tag-like tokens and common fragments produced by split
        text = re.sub(r'<\|[^>]*?\|>', '', text)
        # Also drop standalone fragments likely from tag splits
        text = re.sub(r'(?:analysis|final|assistant)', '', text)
        return text


    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        **kwargs,
    ) -> Union[DeltaMessage, None]:
        # String-based parsing aligned with vLLM style: use tags in text context
        prev = previous_text or ''
        delta = delta_text or ''
        text = prev + delta

        def analysis_open(t: str) -> bool:
            i = t.rfind(self.ANALYSIS)
            if i == -1:
                return False
            j = t.find(self.END, i + len(self.ANALYSIS))
            return j == -1  # no END after last ANALYSIS

        def final_open(t: str) -> bool:
            i1 = t.rfind(self.FINAL1)
            i2 = t.rfind(self.FINAL2)
            i = max(i1, i2)
            return i != -1

        # Case A: analysis is already open from previous_text
        if analysis_open(prev):
            if self.END in delta:
                cut = delta.find(self.END)
                reasoning_part = delta[:cut]
                remainder = delta[cut + len(self.END):]
                # If final marker appears in remainder, only keep content after marker
                if self.FINAL1 in remainder:
                    content_part = remainder.split(self.FINAL1, 1)[1]
                elif self.FINAL2 in remainder:
                    content_part = remainder.split(self.FINAL2, 1)[1]
                else:
                    content_part = remainder or None
                # Drop a trailing END if any
                if content_part and self.END in content_part:
                    content_part = content_part.split(self.END, 1)[0]
                return DeltaMessage(reasoning_content=self._strip_tags(reasoning_part) or None, content=self._strip_tags(content_part) or None)
            else:
                return DeltaMessage(reasoning_content=self._strip_tags(delta) or None)

        # Case B: analysis starts within this delta
        if self.ANALYSIS in delta:
            aft = delta.split(self.ANALYSIS, 1)[1]
            if self.END in aft:
                cut = aft.find(self.END)
                reasoning_part = aft[:cut]
                remainder = aft[cut + len(self.END):]
                if self.FINAL1 in remainder:
                    content_part = remainder.split(self.FINAL1, 1)[1]
                elif self.FINAL2 in remainder:
                    content_part = remainder.split(self.FINAL2, 1)[1]
                else:
                    content_part = remainder or None
                if content_part and self.END in content_part:
                    content_part = content_part.split(self.END, 1)[0]
                return DeltaMessage(reasoning_content=self._strip_tags(reasoning_part) or None, content=self._strip_tags(content_part) or None)
            else:
                return DeltaMessage(reasoning_content=self._strip_tags(aft) or None)

        # Case C: final has started (previously or within delta)
        if final_open(prev) or self.FINAL1 in delta or self.FINAL2 in delta:
            if self.FINAL1 in delta:
                content_part = delta.split(self.FINAL1, 1)[1]
            elif self.FINAL2 in delta:
                content_part = delta.split(self.FINAL2, 1)[1]
            else:
                content_part = delta
            if content_part and self.END in content_part:
                content_part = content_part.split(self.END, 1)[0]
            return DeltaMessage(content=self._strip_tags(content_part) or None)

        # Default: treat as reasoning until markers appear
        return DeltaMessage(reasoning_content=self._strip_tags(delta) or None)

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest, **kwargs):
        text = model_output or ''
        # Extract analysis between ANALYSIS and END
        reasoning = None
        final = None
        if self.ANALYSIS in text and self.END in text:
            a = text.split(self.ANALYSIS, 1)[1]
            reasoning = a.split(self.END, 1)[0]
        # Extract final after FINAL1/FINAL2
        if self.FINAL1 in text:
            final = text.split(self.FINAL1, 1)[1]
        elif self.FINAL2 in text:
            final = text.split(self.FINAL2, 1)[1]
        # Cleanup trailing END if present in final
        if final is not None:
            final = final.split(self.END, 1)[0]
            final = final or None
        # If no tags at all, treat whole as final
        if reasoning is None and final is None and text:
            final = text
        return self._strip_tags(reasoning), self._strip_tags(final)
