"""Exercise the codec against VERL's current CT API, without builder mocks."""

import pytest

from tests.uni_agent.support import FakeProcessor, FakeTokenizer
from uni_agent.gateway.session.codec import MessageCodec
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class _QwenTokenizer(FakeTokenizer):
    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 1000}.get(token)


@pytest.mark.parametrize(
    ("model_type", "tokenizer_path"),
    [("qwen3_5", "local-checkpoint"), ("qwen3_5_moe", "local-checkpoint"), (None, "Qwen/Qwen3.5-9B")],
)
def test_qwen35_builder_preserves_generated_metadata_and_zero_masks_context(model_type, tokenizer_path):
    tokenizer = _QwenTokenizer()
    tokenizer.name_or_path = tokenizer_path
    codec = MessageCodec(tokenizer, hf_model_type=model_type)
    assert isinstance(codec._continuous_token_builder, QwenContinuousTokenBuilder)
    messages = [{"role": "user", "content": "hello"}]
    initial = codec.build_initial_tokens(messages)
    assert initial == tokenizer.apply_chat_template(messages)

    generated = [ord("A"), 1000]
    runtime, mask, logprobs = codec.merge_assistant_tokens(initial, generated, [], [], assistant_logprobs=[-0.25, -0.5])
    previous = messages + [{"role": "assistant", "content": "A"}]
    updated = previous + [{"role": "tool", "content": "result"}]
    merged, mask, logprobs = codec.merge_context_tokens(previous, updated, runtime, mask, logprobs)

    boundary = len(initial) + len(generated)
    assert merged[:boundary] == initial + generated
    assert merged[boundary] == ord("\n")  # Qwen CT inserts the missing turn newline.
    assert tokenizer.decode(merged[boundary:]) == "\ntool:result\nassistant:"
    assert len(mask) == len(logprobs) == len(merged) - len(initial)
    assert mask == [1, 1] + [0] * (len(mask) - 2)
    assert logprobs == [-0.25, -0.5] + [0.0] * (len(logprobs) - 2)


@pytest.mark.parametrize("media_kind", ["image", "video"])
def test_initial_media_uses_processor_and_preserves_processing_options(media_kind):
    processor = FakeProcessor()
    codec = MessageCodec(FakeTokenizer(), processor=processor, mm_processor_kwargs={"max_pixels": 1024})
    messages = [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
    images = ["image://one"] if media_kind == "image" else None
    videos = [("video://one", {"fps": 2}), ("video://two", {"fps": 4})] if media_kind == "video" else None

    token_ids = codec.build_initial_tokens(messages, image_data=images, video_data=videos)
    call = processor.last_processor_call
    assert call["images"] == images
    assert call["videos"] == (["video://one", "video://two"] if videos else None)
    assert call["video_metadata"] == ([{"fps": 2}, {"fps": 4}] if videos else None)
    assert call["extra_kwargs"] == {"max_pixels": 1024}
    assert call["return_tensors"] == "pt"
    assert call["do_sample_frames"] is False
    media_id = processor.image_token_id if images else processor.video_token_id
    assert token_ids.count(media_id) == (1 if images else 2)
    assert token_ids[-len("assistant:") :] == FakeTokenizer().encode("assistant:")
    assert codec.mm_processor_kwargs == {"max_pixels": 1024}


@pytest.mark.parametrize("media_kwargs", [{"image_data": ["image"]}, {"video_data": ["video"]}])
def test_media_without_processor_is_rejected(media_kwargs):
    codec = MessageCodec(FakeTokenizer())
    with pytest.raises(ValueError, match="processor is required"):
        codec.build_initial_tokens([{"role": "user", "content": "describe"}], **media_kwargs)


def test_media_blocks_without_processor_are_not_silently_dropped():
    codec = MessageCodec(FakeTokenizer())
    with pytest.raises(ValueError, match="processor is required"):
        codec.build_initial_tokens(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image://one"}}]}]
        )


def test_replacement_assistant_and_tool_context_remains_untrainable():
    codec = MessageCodec(FakeTokenizer())
    previous = [{"role": "user", "content": "hello"}]
    updated = previous + [
        {"role": "assistant", "content": "replacement"},
        {"role": "tool", "content": "result"},
    ]
    # Session rollback removes the old assistant scaffold before merging context.
    runtime = FakeTokenizer().apply_chat_template(previous, add_generation_prompt=False)
    merged, mask, logprobs = codec.merge_context_tokens(previous, updated, runtime, [], [])
    assert merged == codec.build_initial_tokens(updated)
    assert mask == [0] * (len(merged) - len(runtime))
    assert logprobs == [0.0] * len(mask)
    assert runtime == FakeTokenizer().apply_chat_template(previous, add_generation_prompt=False)


@pytest.mark.parametrize("media_kwargs", [{"image_data": ["image"]}, {"video_data": ["video"]}])
def test_incremental_media_is_rejected_without_replacing_runtime_or_metadata(media_kwargs):
    codec = MessageCodec(FakeTokenizer(), processor=FakeProcessor())
    previous = [{"role": "user", "content": "hello"}]
    updated = previous + [{"role": "tool", "content": "result"}]
    runtime, mask, logprobs = [1, 2, 3], [1], [-0.5]
    with pytest.raises(ValueError, match="does not currently support incremental image or video data"):
        codec.merge_context_tokens(previous, updated, runtime, mask, logprobs, **media_kwargs)
    assert runtime == [1, 2, 3]
    assert mask == [1]
    assert logprobs == [-0.5]
