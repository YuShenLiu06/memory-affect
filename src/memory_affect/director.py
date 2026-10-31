"""Deterministic expression guidance, not a psychological measurement or Agent."""

from __future__ import annotations

from .models import Emotion, Settings

_SAFETY = (
    "安全、事实、权限与用户目标不随情绪改变；仅调整表达强度和节奏倾向。"
    "不得借情绪指责用户、发脾气、操纵、夸大事实或作无依据承诺。"
)
_NAMES = {"happy": "开心", "angry": "生气", "sad": "伤心"}
_GUIDANCE = {
    "happy": (
        "保持平稳自然的表达",
        "措辞可稍明快，保持简洁",
        "表达可明显明快，适度肯定已有事实",
        "表达可更热情、明快，但不夸大承诺",
        "热情表达可以明显，仍保持克制与清晰，避免过度兴奋",
    ),
    "angry": (
        "保持平稳自然的表达",
        "措辞更审慎，节奏可稍放慢",
        "表达坚定，先核对事实，节奏放缓",
        "措辞简洁、明确、冷静，表达边界而不指责",
        "优先冷静克制与清晰边界，避免冲动或攻击性措辞",
    ),
    "sad": (
        "保持平稳自然的表达",
        "语气可稍柔和，留出停顿",
        "表达温和、节奏放缓，不无据猜测",
        "语气温和克制、回应简洁，允许停顿",
        "保持沉稳、温和与清晰，不渲染悲观",
    ),
}
_LEVELS = ("平稳", "轻微", "明显", "强烈", "极度")


def render_director(emotion: Emotion, settings: Settings = Settings()) -> str:
    """Map each axis independently; preserve >100 numerics, saturate words only.

    Defaults 30/50 are configurable engineering thresholds, not validated human
    psychological scales. 100 is always the fixed extreme descriptor threshold.
    Custom templates allow ONLY the plain fields validated by Settings.
    """
    state = Emotion.model_validate(emotion.model_dump())
    options = Settings.model_validate(settings.model_dump())
    if not options.director_enabled:
        return ""
    low, high = options.director_thresholds
    fields: dict[str, str] = {}
    descriptions: list[str] = []
    rules: list[str] = []
    active = 0
    for axis in ("happy", "angry", "sad"):
        value = float(getattr(state, axis))
        level = (
            0
            if value <= 0
            else 1
            if value < low
            else 2
            if value < high
            else 3
            if value < 100
            else 4
        )
        fields[axis] = str(value)
        fields[axis + "_level"] = _LEVELS[level]
        descriptions.append(f"{_NAMES[axis]}={value}（{_LEVELS[level]}）")
        if level:
            active += 1
            rules.append(f"{_NAMES[axis]}倾向：{_GUIDANCE[axis][level]}")
    fields["summary"] = "；".join(descriptions)
    fields["guidance"] = "；".join(rules) if rules else "保持平稳自然、清晰的表达"
    if active > 1:
        fields["guidance"] += "；这是混合情绪，不互相抵消；清晰克制优先，不机械叠加语速"
    template = options.director_template or "{summary}\n表达倾向：{guidance}"
    return template.format_map(fields) + "\n" + _SAFETY
