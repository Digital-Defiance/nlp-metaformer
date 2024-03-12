from typing import Literal

AttentionMechanisms = Literal[
    "quadratic",
    "metric", 
    "scaled_dot_product"
    "identity",
    "average_pooling",
]




SourceExecutable = Literal[
    "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.2/llm-voice-chat",
    "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.1/llm-voice-chat",
    "/__w/llm-voice-chat/llm-voice-chat/target/debug/llm-voice-chat",
]


DEFAULT_ATTENTION_MECHANISM: AttentionMechanisms = "quadratic"
SAVE_PATH: str = "output.safetensors"
DEV_RUST_BINARY: str = "/__w/llm-voice-chat/llm-voice-chat/target/debug/llm-voice-chat"

