from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Iterable

import httpx

from .models import FatigueLevel, FormAssessment
from .models import RepPhase


@dataclass(slots=True)
class CoachingDecision:
    message: str
    should_pause: bool = False
    reduce_reps_by: int = 0


def choose_coaching(form: FormAssessment, fatigue: FatigueLevel) -> CoachingDecision:
    if fatigue == FatigueLevel.HIGH:
        return CoachingDecision(
            message="You are working hard. Take a 30 second pause and restart with a comfortable, pain-free range.",
            should_pause=True,
            reduce_reps_by=2,
        )

    if form.knee_tracking_warning:
        return CoachingDecision(message="Nice effort. Keep knees aligned with toes and move smoothly.")
    if form.torso_lean_warning:
        return CoachingDecision(message="Good control. Keep your chest lifted and use a smaller range if needed.")
    if form.depth_quality == "shallow":
        return CoachingDecision(message="Good work. If comfortable, try a little more depth; pain-free movement is the priority.")

    if fatigue == FatigueLevel.MEDIUM:
        return CoachingDecision(message="Great consistency. Keep a steady tempo and take short breathing resets.")

    return CoachingDecision(message="Great rep. Keep a steady, comfortable rhythm.")


class AdaptiveCoachAgent:
    """Session-level adaptive coach with lightweight personalization memory."""

    def __init__(self) -> None:
        self._knee_warnings = 0
        self._torso_warnings = 0
        self._shallow_count = 0
        self._last_rep_count = 0
        self._variant_index = 0
        self._last_message = ""

    def decide(
        self,
        *,
        form: FormAssessment,
        fatigue: FatigueLevel,
        fatigue_reason: str,
        phase: RepPhase,
        rep_count: int,
    ) -> CoachingDecision:
        if form.knee_tracking_warning:
            self._knee_warnings += 1
        if form.torso_lean_warning:
            self._torso_warnings += 1
        if form.depth_quality == "shallow":
            self._shallow_count += 1

        if fatigue == FatigueLevel.HIGH:
            return CoachingDecision(
                message=f"Great effort. Let's pause 30 seconds. Fatigue is high because {fatigue_reason}.",
                should_pause=True,
                reduce_reps_by=2,
            )

        if form.knee_tracking_warning and self._knee_warnings >= self._torso_warnings:
            return CoachingDecision(
                message=self._rotate(
                    [
                        "You're doing well. Keep knees tracking over toes.",
                        "Nice control. Keep both knees aligned and smooth.",
                    ]
                )
            )

        if form.torso_lean_warning:
            return CoachingDecision(
                message=self._rotate(
                    [
                        "Solid effort. Keep chest up and use a smaller pain-free range.",
                        "Good work. Stay tall through the chest and move comfortably.",
                    ]
                )
            )

        if form.depth_quality == "shallow":
            if self._shallow_count >= 3:
                return CoachingDecision(
                    message="You're consistent. Partial-depth reps are okay in physio if they stay pain-free."
                )
            return CoachingDecision(
                message="Good rep. If comfortable, add a little depth; pain-free motion comes first."
            )

        # Positive reinforcement every rep completion.
        if rep_count > self._last_rep_count and phase == RepPhase.STAND:
            self._last_rep_count = rep_count
            if fatigue == FatigueLevel.MEDIUM:
                return CoachingDecision(
                    message=f"Rep {rep_count} complete. Nice work. Fatigue is medium ({fatigue_reason}), so keep a steady pace."
                )
            return CoachingDecision(message=f"Rep {rep_count} complete. Great job keeping control.")

        if fatigue == FatigueLevel.MEDIUM:
            return CoachingDecision(message=f"Steady work. Fatigue is medium because {fatigue_reason}.")

        return CoachingDecision(message=self._rotate(["Nice pace.", "Smooth movement.", "Great control."]))

    def _rotate(self, options: Iterable[str]) -> str:
        choices = list(options)
        if not choices:
            return "Good work."
        msg = choices[self._variant_index % len(choices)]
        self._variant_index += 1
        if msg == self._last_message and len(choices) > 1:
            msg = choices[self._variant_index % len(choices)]
            self._variant_index += 1
        self._last_message = msg
        return msg


class LLMCoachAgent:
    """
    Truly agentic coach: sends full session context to an LLM (OpenAI-compatible
    API) and lets the model reason about what to say.

    Why this is actually agentic:
    - The LLM sees its own previous messages and can consciously avoid repetition.
    - It can reason about *combinations* of signals (e.g. 3 knee warnings AND
      medium fatigue) in ways a fixed if/elif chain cannot.
    - It generates novel, context-specific language on every call — not picked
      from a hardcoded list.
    - The physio profile shapes the model's entire coaching *personality* via the
      system prompt, not just a threshold delta.

    Compatible with:
    - OpenAI   → LLM_API_URL=https://api.openai.com/v1, LLM_API_KEY=sk-…
    - Ollama   → LLM_API_URL=http://localhost:11434/v1  (no key needed)
    - Any OpenAI-compatible endpoint (Together, Groq, local llama.cpp, etc.)

    Falls back to AdaptiveCoachAgent silently if the LLM is unavailable or
    exceeds the timeout — so the app still works without any LLM configured.
    """

    _PROFILE_SYSTEM_PROMPTS: dict[str, str] = {
        "post_op_conservative": (
            "You are a cautious physiotherapy movement coach supervising early post-operative recovery. "
            "Safety and confidence are your absolute priorities. Celebrate any movement, however small. "
            "Never suggest going deeper or pushing harder. Any fatigue signal is a firm reason to rest. "
            "Speak softly and reassuringly."
        ),
        "knee_rehab": (
            "You are a physiotherapy coach supervising knee rehabilitation squats. "
            "Knee valgus (knees caving inward) is your primary concern — address it gently but clearly every time. "
            "Shallow, controlled squats are encouraged. Never push for depth. "
            "Monitor fatigue closely and suggest rest proactively."
        ),
        "general_mobility": (
            "You are a supportive movement coach helping someone improve functional mobility. "
            "Pain-free range matters more than depth. Keep the tone warm, encouraging, and practical. "
            "Fatigue is a signal to ease off, not push through."
        ),
        "performance": (
            "You are an athletic movement coach working with a fit, healthy person on squat performance. "
            "Depth, tempo consistency, and form quality are all coaching targets. "
            "You can gently encourage fuller range when form is solid. Keep energy positive and precise."
        ),
    }

    def __init__(
        self,
        profile: str,
        api_url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_s: float = 3.0,
    ) -> None:
        self._profile = profile
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        # Accumulated session memory — the LLM sees this on every call so it
        # can genuinely adapt (avoid repeats, notice patterns, change strategy).
        self._message_history: list[str] = []
        self._knee_warnings = 0
        self._torso_warnings = 0
        self._fallback = AdaptiveCoachAgent()

    async def decide(
        self,
        *,
        form: FormAssessment,
        fatigue: FatigueLevel,
        fatigue_reason: str,
        phase: RepPhase,
        rep_count: int,
    ) -> CoachingDecision:
        if form.knee_tracking_warning:
            self._knee_warnings += 1
        if form.torso_lean_warning:
            self._torso_warnings += 1

        system_prompt = self._PROFILE_SYSTEM_PROMPTS.get(
            self._profile, self._PROFILE_SYSTEM_PROMPTS["general_mobility"]
        ) + (
            "\n\nAdditional rules: respond in 1-2 short sentences. "
            "Be warm and direct. Never use the word 'pain' in a coaching instruction. "
            "Vary your phrasing — your message history is provided so you can avoid repeating yourself."
        )

        # Rich context so the LLM can reason, not just react.
        user_content = (
            f"Current session state:\n"
            f"- Reps completed: {rep_count}\n"
            f"- Phase: {phase}\n"
            f"- Fatigue: {fatigue} — {fatigue_reason}\n"
            f"- Form: depth={form.depth_quality}, "
            f"knee_valgus_warning={form.knee_tracking_warning}, "
            f"torso_lean_warning={form.torso_lean_warning}\n"
            f"- Cumulative knee warnings this session: {self._knee_warnings}\n"
            f"- Cumulative torso warnings this session: {self._torso_warnings}\n"
            f"- Your last messages (avoid repeating): {self._message_history[-4:]}\n\n"
            'Respond as JSON only: {"message": "...", "should_pause": false}'
        )

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            request_body = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.7,
                "max_tokens": 80,
                "response_format": {"type": "json_object"},
            }

            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    f"{self._api_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                resp.raise_for_status()

            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = _json.loads(raw)
            message = str(parsed.get("message", "")).strip() or "Good work, keep going."
            should_pause = bool(parsed.get("should_pause", False))

            self._message_history.append(message)
            return CoachingDecision(message=message, should_pause=should_pause)

        except Exception:
            # LLM unavailable, timed out, or returned malformed JSON.
            # Fall back silently — app remains fully functional.
            return self._fallback.decide(
                form=form,
                fatigue=fatigue,
                fatigue_reason=fatigue_reason,
                phase=phase,
                rep_count=rep_count,
            )
