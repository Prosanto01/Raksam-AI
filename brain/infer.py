"""
Safe inference for Raksam Brain V3 (RB3).

Created by: Prosanto Raksam

No external AI model or AI runtime is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from brain.model import BrainConfig, BrainModel
from brain.tokenizer.tokenizer import CharTokenizer
from brain.dataset import action_prompt, chat_prompt


MODEL_NAME = "Raksam Brain V3 (RB3)"
CREATOR = "Prosanto Raksam"


class Brain:
    """
    Inference wrapper for Raksam Brain V3.

    Handles:
    - checkpoint loading
    - tokenizer/model compatibility
    - safe text generation
    - chat generation
    - action generation
    - malformed-output protection
    """

    def __init__(
        self,
        checkpoint_path: str = "brain/checkpoints/latest.pt",
        tokenizer_path: str = "brain/tokenizer/tokenizer.json",
        device: str | None = None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = CharTokenizer(
            tokenizer_path
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(ckpt, dict):
            raise ValueError(
                "Invalid Raksam Brain checkpoint."
            )

        cfg_data = dict(
            ckpt.get("config", {})
        )

        cfg_data["vocab_size"] = int(
            cfg_data.get(
                "vocab_size",
                self.tokenizer.vocab_size,
            )
        )

        # ----------------------------------------------------
        # Tokenizer / checkpoint safety check
        # ----------------------------------------------------

        if (
            cfg_data["vocab_size"]
            != self.tokenizer.vocab_size
        ):
            raise ValueError(
                "Checkpoint/tokenizer vocabulary mismatch. "
                f"Checkpoint={cfg_data['vocab_size']} "
                f"Tokenizer={self.tokenizer.vocab_size}. "
                "Retrain a fresh checkpoint using the current tokenizer."
            )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        cfg = BrainConfig(
            **cfg_data
        )

        self.model = BrainModel(
            cfg
        ).to(self.device)

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        state = ckpt.get(
            "model_state"
        )

        if state is None:
            raise ValueError(
                "Checkpoint does not contain model_state."
            )

        self.model.load_state_dict(
            state,
            strict=True,
        )

        self.model.eval()

        self.config = cfg

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.model_name = ckpt.get(
            "model_name",
            MODEL_NAME,
        )

        self.created_by = ckpt.get(
            "created_by",
            CREATOR,
        )

        # Small local retrieval index built only from Raksam's own approved
        # conversation training data. This is not another AI model; it gives
        # the tiny from-scratch neural brain a reliable memory for learned facts
        # while it continues improving through neural training.
        self.learned_answers = {}
        self._load_learned_answers()

    @staticmethod
    def _norm_query(text: str) -> str:
        text = str(text or "").lower().strip()
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    def _load_learned_answers(self):
        path = Path("data/conversations/chat_training.jsonl")
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    q = self._norm_query(row.get("user", ""))
                    a = str(row.get("assistant", "")).strip()
                    if q and a:
                        self.learned_answers[q] = a
        except Exception:
            pass

    def _known_query_score(self, user_text: str) -> float:
        q = self._norm_query(user_text)
        words = set(q.split())
        if not words:
            return 0.0
        best = 0.0
        for known in self.learned_answers:
            kw = set(known.split())
            if not kw:
                continue
            score = len(words & kw) / max(1, len(words | kw))
            best = max(best, score)
        return best

    def _remembered_answer(self, user_text: str) -> str | None:
        q = self._norm_query(user_text)
        if not q:
            return None
        exact = self.learned_answers.get(q)
        if exact:
            return exact
        # Conservative fuzzy match: only reuse a learned answer when the
        # query words overlap strongly and the query is not too short.
        words = set(q.split())
        if len(words) < 3:
            return None
        best_score, best_answer = 0.0, None
        for known, answer in self.learned_answers.items():
            kw = set(known.split())
            if not kw:
                continue
            score = len(words & kw) / max(1, len(words | kw))
            if score > best_score:
                best_score, best_answer = score, answer
        return best_answer if best_score >= 0.90 else None

    @staticmethod
    def _action_heuristic(prompt_text: str) -> dict | None:
        """Reliable parser for simple action intents before neural planning."""
        goal_match = re.search(r"GOAL:\s*(.*?)(?:\nDEVICE:|\nCURRENT_OBSERVATION:|$)", prompt_text, re.I | re.S)
        goal = (goal_match.group(1).strip() if goal_match else prompt_text).strip()
        g = goal.lower()
        device = "my_android" if "android" in g or "phone" in g else "this_pc"
        # Compound goals must remain in AgentLoop.  Returning a later action
        # here used to skip opening/focusing the target and defeated replan.
        m = re.match(r"^(?:type|write)\s+(.+)$", goal, re.I)
        if m: return {"action":"type","target":m.group(1).strip(),"device":device,"reason":"simple typing intent"}
        m = re.match(r"^(?:press|hit)\s+(.+)$", goal, re.I)
        if m: return {"action":"key","target":m.group(1).strip(),"device":device,"reason":"simple key intent"}
        m = re.match(r"^(?:click|tap)\s+(?:the\s+)?(.+)$", goal, re.I)
        if m: return {"action":"click","target":m.group(1).strip(),"device":device,"reason":"simple UI intent"}
        m = re.match(r"^(?:open|launch|start)\s+(?:the\s+)?(.+)$", goal, re.I)
        if m:
            target=m.group(1).strip()
            if target.lower().startswith(("http://","https://")):
                return {"action":"open_url","target":target,"device":device,"reason":"URL intent"}
            return {"action":"launch_app","target":target,"device":device,"reason":"application intent"}
        return None

    # ========================================================
    # INTERNAL GENERATION
    # ========================================================

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.68,
        min_new_tokens: int = 8,
    ) -> str:
        """
        Generate text from a prompt.

        EOS is suppressed during the first min_new_tokens
        generated tokens so the model has time to produce
        a meaningful response.
        """

        # ----------------------------------------------------
        # Encode prompt
        # ----------------------------------------------------

        ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )

        max_context = max(
            2,
            int(self.config.block_size),
        )

        # ----------------------------------------------------
        # Keep context inside model limit.
        # ----------------------------------------------------

        if len(ids) >= max_context:
            ids = ids[
                -(max_context - 1):
            ]

        # ----------------------------------------------------
        # Available generation space.
        # ----------------------------------------------------

        available_tokens = (
            max_context - len(ids)
        )

        if available_tokens <= 0:
            return ""

        max_new_tokens = min(
            int(max_new_tokens),
            available_tokens,
        )

        if max_new_tokens <= 0:
            return ""

        min_new_tokens = min(
            max(0, int(min_new_tokens)),
            max_new_tokens,
        )

        # ----------------------------------------------------
        # Input tensor
        # ----------------------------------------------------

        idx = torch.tensor(
            [ids],
            dtype=torch.long,
            device=self.device,
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.inference_mode():

            out = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.08,
                stop_ids=[
                    self.tokenizer.eos_id
                ],
                min_new_tokens=min_new_tokens,
            )

        # ----------------------------------------------------
        # Decode only newly generated tokens.
        # ----------------------------------------------------

        generated_ids = out[
            0,
            len(ids):,
        ].tolist()

        text = self.tokenizer.decode(
            generated_ids
        )

        return text.strip()

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Remove control characters and excessive blank lines.
        """

        text = str(
            text or ""
        ).replace(
            "\x00",
            "",
        )

        # Remove ASCII control characters except newline/tab.
        text = re.sub(
            r"[\u0001-\u0008\u000b\u000c\u000e-\u001f]",
            "",
            text,
        )

        # Limit excessive blank lines.
        text = re.sub(
            r"\n{4,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ========================================================
    # TEXT VALIDATION
    # ========================================================

    @staticmethod
    def _valid_text(
        text: str,
    ) -> bool:
        """
        Detect obviously broken model output.

        This does not determine factual correctness.
        """

        if not text:
            return False

        text = text.strip()

        if len(text) < 2:
            return False

        # Too many replacement characters indicate malformed
        # UTF-8 output.
        if text.count("\ufffd") > max(
            2,
            len(text) // 20,
        ):
            return False

        # Detect extreme character repetition.
        if len(text) >= 12:

            unique = len(
                set(text)
            )

            if unique <= 3:
                return False

        # Detect obvious broken JSON / escaped output.
        if re.search(
            r"(?:\{){8,}"
            r"|(?:\}){8,}"
            r"|(?:\\x[0-9a-fA-F]){6,}",
            text,
        ):
            return False

        return True

    # ========================================================
    # REMOVE ROLE CONTINUATION
    # ========================================================

    @staticmethod
    def _remove_role_continuation(
        text: str,
    ) -> str:
        """
        Prevent the model from continuing into another
        conversation role.
        """

        pattern = (
            r"\n(?:"
            r"USER"
            r"|ASSISTANT"
            r"|<CHAT>"
            r"|<TASK>"
            r"|<ASSISTANT_ACTION>"
            r"|<KNOWLEDGE>"
            r"|<REFLECTION>"
            r")\s*:?"
        )

        text = re.split(
            pattern,
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        return text.strip()

    @staticmethod
    def _identity_answer(text: str) -> str | None:
        q=re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
        if q in {
            "what is your name", "whats your name", "what s your name",
            "what is ur name", "whats ur name", "your name", "name",
            "tell me your name", "what s ur name",
        }:
            return "My name is Raksam AI."
        if q in {
            "who created you", "who made you", "who is your creator",
            "who created raksam ai", "who is your creator name",
        }:
            return "I was created by Prosanto Raksam."
        if q in {"how are you", "how are u", "how r u", "how are you doing", "are you okay"}:
            return "I'm doing well and ready to help."
        if q in {"hello", "hi", "hey", "hello raksam", "hi raksam", "hey raksam"}:
            return "Hello! I'm Raksam. How can I help you?"
        return None

    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        user_text: str,
        history: list | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate a conversational answer.
        """

        user_text = str(
            user_text or ""
        ).strip()

        if not user_text:
            return (
                "Please tell me what you would like help with."
            )

        identity = self._identity_answer(user_text)
        if identity:
            return identity

        # A high-confidence learned answer is preferable to asking a tiny
        # under-trained neural model to regenerate an exact memorized fact.
        lookup_text = user_text.split("\n\n[CONTEXT]", 1)[0].strip()
        remembered = self._remembered_answer(lookup_text)
        if remembered:
            return remembered

        # The packaged neural model is intentionally small and CPU-trainable.
        # If the query is outside the learned curriculum, do not hallucinate a
        # plausible-looking answer; return an explicit learning fallback.
        if self._known_query_score(lookup_text) < 0.34:
            return (
                "I'm still learning. I don't have a reliable answer for that yet."
            )

        # ----------------------------------------------------
        # Build chat prompt.
        # ----------------------------------------------------

        prompt = chat_prompt(
            user_text,
            history or [],
        )

        # ----------------------------------------------------
        # Generate.
        #
        # Minimum 8 generated tokens before EOS.
        # ----------------------------------------------------

        text = self._generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.68,
            min_new_tokens=8,
        )

        # ----------------------------------------------------
        # Clean generated text.
        # ----------------------------------------------------

        text = self._clean_text(
            text
        )

        # ----------------------------------------------------
        # Remove accidental role continuation.
        # ----------------------------------------------------

        text = self._remove_role_continuation(
            text
        )

        # ----------------------------------------------------
        # Validate.
        # ----------------------------------------------------

        # Do not recycle a memorized answer for an unrelated question.
        if text in self.learned_answers.values() and not remembered:
            return (
                "I'm still learning. "
                "I don't have a reliable answer for that yet."
            )

        if not self._valid_text(
            text
        ):
            print(
                "DEBUG: Raksam generated "
                "an invalid/empty answer:",
                repr(text),
            )

            return (
                "I'm still learning. "
                "I don't have a reliable answer for that yet."
            )

        return text

    # ========================================================
    # ACTION
    # ========================================================

    def act(
        self,
        prompt_text: str,
        max_new_tokens: int = 64,
    ) -> dict:
        """
        Generate an action request.

        Action generation uses a short minimum token count
        because action JSON should be able to terminate quickly.
        """

        prompt_text = str(
            prompt_text or ""
        ).strip()

        if not prompt_text:
            return {
                "action": "wait",
                "target": "1",
                "reason": (
                    "No action request was provided."
                ),
            }

        heuristic = self._action_heuristic(prompt_text)
        if heuristic:
            return heuristic

        # ----------------------------------------------------
        # Build action prompt.
        # ----------------------------------------------------

        prompt = action_prompt(
            prompt_text
        )

        # ----------------------------------------------------
        # Generate action.
        # ----------------------------------------------------

        generated = self._generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.18,
            min_new_tokens=1,
        )

        return self._parse_action(
            generated
        )

    # ========================================================
    # ACTION JSON PARSER
    # ========================================================

    @staticmethod
    def _parse_action(
        text: str,
    ) -> dict:
        """
        Safely extract an action JSON object.

        Malformed output becomes a safe wait action.
        """

        text = str(
            text or ""
        ).strip()

        candidates: list[str] = []

        # ----------------------------------------------------
        # Fenced JSON.
        # ----------------------------------------------------

        fenced = re.search(
            r"```(?:json)?\s*"
            r"(\{.*?\})"
            r"\s*```",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if fenced:
            candidates.append(
                fenced.group(1)
            )

        # ----------------------------------------------------
        # Ordinary JSON objects.
        # ----------------------------------------------------

        candidates.extend(
            re.findall(
                r"\{[^{}]{1,2000}\}",
                text,
                re.DOTALL,
            )
        )

        # ----------------------------------------------------
        # Parse candidates.
        # ----------------------------------------------------

        for candidate in candidates:

            try:
                value = json.loads(
                    candidate
                )

            except json.JSONDecodeError:
                continue

            if not isinstance(
                value,
                dict,
            ):
                continue

            action = value.get(
                "action"
            )

            if not isinstance(
                action,
                str,
            ):
                continue

            action = action.strip()

            if not action:
                continue

            return value

        # ----------------------------------------------------
        # Safe fallback.
        # ----------------------------------------------------

        return {
            "action": "wait",
            "target": "1",
            "reason": (
                "Raksam could not produce a valid action; "
                "execution paused safely."
            ),
        }


# ============================================================
# SIMPLE DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        MODEL_NAME
    )

    print(
        f"Created by: {CREATOR}"
    )

    brain = Brain()

    print(
        f"Device: {brain.device}"
    )

    print(
        f"Parameters: "
        f"{brain.model.num_parameters():,}"
    )

    while True:

        try:
            user = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print(
                "\nExiting."
            )
            break

        if not user:
            continue

        if user.lower() in {
            "quit",
            "exit",
            "/quit",
        }:
            break

        answer = brain.chat(
            user
        )

        print(
            "Raksam:",
            answer,
        )
