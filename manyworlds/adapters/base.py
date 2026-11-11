"""Capability interfaces. Every outside service sits behind one of these.

Adding a model or vendor means writing one adapter that satisfies an interface and
registering a Capability for it; the router then chooses among capabilities using
observed cost per accepted output (see manyworlds.adapters.registry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

INTERFACES = (
    "DataSource", "LLM", "ImageGen", "ImageEdit", "VideoGen",
    "LipSync", "TTS", "VisionJudge", "Publisher",
)


@dataclass(frozen=True)
class Capability:
    """A routable production option, e.g. 'Z-Image Turbo on SaladCloud 4090s'."""

    name: str
    interface: str
    job_kind: str | None = None        # set for capabilities served by GPU workers
    hosted_usd_per_call: float | None = None  # set for hosted APIs billed per call
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.interface not in INTERFACES:
            raise ValueError(f"unknown interface {self.interface!r}")


@dataclass
class ImageRequest:
    prompt: str
    n: int = 4
    width: int = 896
    height: int = 1152
    seed: int | None = None
    lora_uri: str | None = None
    lora_scale: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoRequest:
    image_uri: str
    prompt: str
    width: int = 480
    height: int = 832
    frames: int = 81
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechRequest:
    text: str
    voice: str
    speed: float = 1.0


@dataclass
class Generated:
    """What a production call returns: stored asset ids plus their provenance."""

    asset_ids: list[Any]
    job_id: Any = None
    usd: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageGen(Protocol):
    def generate(self, req: ImageRequest, *, creator_id=None) -> Generated: ...


@runtime_checkable
class VideoGen(Protocol):
    def generate(self, req: VideoRequest, *, creator_id=None) -> Generated: ...


@runtime_checkable
class TTS(Protocol):
    def speak(self, req: SpeechRequest, *, creator_id=None) -> Generated: ...


@runtime_checkable
class LLM(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]], **kw) -> str: ...


@runtime_checkable
class VisionJudge(Protocol):
    def judge(self, asset_uri: str, rubric: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class Publisher(Protocol):
    def publish(self, post: dict[str, Any]) -> str: ...


@runtime_checkable
class DataSource(Protocol):
    def fetch(self, query: dict[str, Any]) -> list[dict[str, Any]]: ...
