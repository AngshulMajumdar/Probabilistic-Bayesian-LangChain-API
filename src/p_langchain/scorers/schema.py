# p_langchain/scorers/schema.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

from p_langchain.core.types import Hypothesis, TraceEvent
from .base import Scorer, ScoreResult


def _try_parse_json(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        obj = json.loads(text)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def _is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(t, str) for t in x)


def _clip01(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0 or v > 1.0:
        return None
    return v


def _pydantic_model_validate(pydantic_model: Any, obj: Any) -> Tuple[bool, Optional[str]]:
    """
    Pydantic v2: model_validate
    Pydantic v1: parse_obj
    """
    try:
        if hasattr(pydantic_model, "model_validate"):
            pydantic_model.model_validate(obj)  # v2
            return True, None
        if hasattr(pydantic_model, "parse_obj"):
            pydantic_model.parse_obj(obj)  # v1
            return True, None
        return False, "pydantic model has no validator method"
    except Exception as e:
        return False, str(e)


@dataclass
class JsonSchemaScorer(Scorer):
    """
    Scores whether hypothesis.state[text_key] is:
      (1) valid JSON
      (2) matches a lightweight schema (dict with required keys/types)
      (3) optionally passes a Pydantic model, if provided

    This is dependency-free by default, but can become strict with Pydantic automatically.

    Required schema (default):
      - task: str
      - steps: list[str]
      - confidence: number in [0,1]
    """
    text_key: str = "llm_text"
    parsed_key: str = "parsed_json"
    name: str = "schema"

    # log-space deltas (executor adds these to logw)
    valid_reward: float = 4.0
    invalid_penalty: float = -8.0

    # Extra shaping: make demo pick a clear best
    missing_key_penalty: float = -3.0
    wrong_type_penalty: float = -3.0
    extra_key_penalty: float = -0.5
    confidence_bonus: float = 1.0  # if confidence is valid [0,1]

    # required schema keys/types
    required_keys: Tuple[str, ...] = ("task", "steps", "confidence")

    # Optional: strict validation with a Pydantic model (v1 or v2)
    pydantic_model: Optional[Type[Any]] = None

    # If True, and pydantic is installed + pydantic_model is None, we won't auto-create a model.
    # (We keep it simple: user supplies a model if they want strictness beyond lightweight checks.)
    auto_detect_pydantic: bool = True

    def score(self, h: Hypothesis) -> ScoreResult:
        raw = h.state.get(self.text_key, "")
        raw = raw if isinstance(raw, str) else str(raw)

        ok, obj, err = _try_parse_json(raw)
        if not ok:
            ev = TraceEvent(
                kind=f"score.{self.name}.invalid_json",
                message="invalid json",
                data={"error": err},
            )
            return ScoreResult(
                score_delta=self.invalid_penalty,
                meta={"valid_json": False, "error": err},
                event=ev,
            )

        meta: Dict[str, Any] = {"valid_json": True, "parsed": obj}
        score = float(self.valid_reward)

        # Lightweight schema validation
        if not isinstance(obj, dict):
            ev = TraceEvent(
                kind=f"score.{self.name}.wrong_top_level",
                message="json ok but top-level not dict",
                data={"top_level_type": type(obj).__name__},
            )
            return ScoreResult(
                score_delta=self.invalid_penalty,
                meta={"valid_json": True, "schema_ok": False, "error": "top_level_not_dict"},
                event=ev,
            )

        # Check required keys
        missing = [k for k in self.required_keys if k not in obj]
        if missing:
            score += self.missing_key_penalty * len(missing)
            meta["missing_keys"] = missing

        # Type checks for present keys
        if "task" in obj and not isinstance(obj["task"], str):
            score += self.wrong_type_penalty
            meta["task_type"] = type(obj["task"]).__name__

        if "steps" in obj and not _is_str_list(obj["steps"]):
            score += self.wrong_type_penalty
            meta["steps_type"] = type(obj["steps"]).__name__

        conf_ok = None
        if "confidence" in obj:
            conf_ok = _clip01(obj["confidence"])
            if conf_ok is None:
                score += self.wrong_type_penalty
                meta["confidence_error"] = "not_in_[0,1]"
            else:
                score += self.confidence_bonus
                meta["confidence_value"] = conf_ok

        # Penalize extra keys slightly (keeps output tight)
        extra = [k for k in obj.keys() if k not in self.required_keys]
        if extra:
            score += self.extra_key_penalty * len(extra)
            meta["extra_keys"] = extra

        # Determine lightweight schema_ok
        schema_ok = (len(missing) == 0) and ("task" in obj and isinstance(obj["task"], str)) and \
                    ("steps" in obj and _is_str_list(obj["steps"])) and \
                    ("confidence" in obj and _clip01(obj["confidence"]) is not None)

        meta["schema_ok"] = schema_ok

        # Optional Pydantic validation if provided
        if self.pydantic_model is not None:
            p_ok, p_err = _pydantic_model_validate(self.pydantic_model, obj)
            meta["pydantic_ok"] = p_ok
            if not p_ok:
                # Penalize strongly: schema mismatch
                score += self.invalid_penalty
                ev = TraceEvent(
                    kind=f"score.{self.name}.pydantic_fail",
                    message="json ok but pydantic schema invalid",
                    data={"error": p_err},
                )
                return ScoreResult(
                    score_delta=float(score),
                    meta={**meta, "error": p_err},
                    event=ev,
                )

        ev = TraceEvent(
            kind=f"score.{self.name}.checked",
            message="json parsed and schema checked",
            data={
                "schema_ok": schema_ok,
                "score_delta": score,
                "missing": missing,
                "extra": extra,
            },
        )
        return ScoreResult(score_delta=float(score), meta=meta, event=ev)
