import json
import logging
import os
import re
import time

from agent_modules.util import load_json
from autogen_core.models import ChatCompletionClient, UserMessage, SystemMessage


# ── Per-module LLM logging ───────────────────────────────────────────────

_LLM_LOG_DIR: str | None = None
_MODULE_LOGGERS: dict[str, logging.Logger] = {}

# (regex, module-file-name). First match wins; ordered most-specific first.
_MODULE_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^Agent .* on_messages"),  "action_selection"),
    (re.compile(r"^Agent .* rl_comm"),      "action_selection_comm"),
    (re.compile(r"^Auto Curriculum"),       "auto_curriculum"),
    (re.compile(r"^Belief System"),         "belief_system"),
    (re.compile(r"^Critic"),                "critic"),
    (re.compile(r"^EpisodicMemoryManager"), "episodic_memory"),
    (re.compile(r"^SocialModule"),          "social_module"),
    (re.compile(r"^SkillManager"),          "skill_manager"),
)


def setup_llm_logging(log_dir) -> None:
    """Enable per-module LLM log files under ``log_dir``.

    Called once from the main entry point (after RunPaths is built). Safe to
    call multiple times — subsequent calls re-target the directory but the
    cached loggers keep their existing handlers attached, so the old files
    won't go silent mid-run.
    """
    global _LLM_LOG_DIR
    _LLM_LOG_DIR = str(log_dir)
    os.makedirs(_LLM_LOG_DIR, exist_ok=True)


def _module_name(log_prefix: str) -> str:
    for pattern, name in _MODULE_PATTERNS:
        if pattern.search(log_prefix):
            return name
    return "llm_other"


def _get_module_logger(log_prefix: str) -> logging.Logger:
    name = _module_name(log_prefix)
    cached = _MODULE_LOGGERS.get(name)
    if cached is not None:
        return cached
    logger = logging.getLogger(f"wiredtogether.llm.{name}")
    logger.setLevel(logging.INFO)
    if _LLM_LOG_DIR:
        path = os.path.join(_LLM_LOG_DIR, f"{name}.log")
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        # Stop these from also flooding the root run log.
        logger.propagate = False
    _MODULE_LOGGERS[name] = logger
    return logger


# ── LLM call ─────────────────────────────────────────────────────────────


async def llm_call(
    model_client: ChatCompletionClient,
    user_prompt,
    cancellation_token,
    system_prompt=None,
    frame=None,
    parse_check=None,
    parser=None,
    retry_count=0,
    log_prefix="LLM call",
    **kwargs,
):
    mod_logger = _get_module_logger(log_prefix)

    # if failed more than 5 times, return empty fallback instead of crashing
    if retry_count > 5:
        mod_logger.error("%s Too many retries (6), returning empty response", log_prefix)
        return {}

    # Build prompt
    try:
        filled_user_prompt = user_prompt.format(**kwargs)
    except KeyError as e:
        mod_logger.error(
            "%s Missing prompt placeholder %s, using raw prompt", log_prefix, e
        )
        filled_user_prompt = user_prompt
    if frame is None:
        user_message = UserMessage(content=[filled_user_prompt], source="user")
    else:
        user_message = UserMessage(content=[filled_user_prompt, frame], source="user")

    if system_prompt is not None:
        full_prompt = [SystemMessage(content=system_prompt), user_message]
    else:
        full_prompt = [user_message]

    # Metadata only — the prompt body is intentionally NOT logged.
    mod_logger.info(
        "%s call: sys_chars=%d user_chars=%d frame=%s kwargs=%s retry=%d",
        log_prefix,
        len(system_prompt) if system_prompt else 0,
        len(filled_user_prompt) if filled_user_prompt else 0,
        frame is not None,
        sorted(kwargs.keys()),
        retry_count,
    )

    # call model
    try:
        response = await model_client.create(
            full_prompt, cancellation_token=cancellation_token
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        error_str = str(e)
        mod_logger.error(
            "%s Error calling LLM (attempt %d): %s",
            log_prefix, retry_count + 1, error_str[:300],
        )
        time.sleep(1)
        return await llm_call(
            model_client=model_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cancellation_token=cancellation_token,
            frame=frame,
            retry_count=retry_count + 1,
            log_prefix=log_prefix,
            **kwargs,
            )

    mod_logger.info("%s Response: %s", log_prefix, response.content)

    # parse response
    try:
        content = (parser or load_json)(response.content)
        if not content:
            raise ValueError("Empty JSON response")
        if parse_check is not None:
            content = parse_check(content)
        return content
    except (json.JSONDecodeError, AssertionError, KeyError, ValueError) as e:
        mod_logger.error(
            "%s Error parsing response (attempt %d): %s: %s",
            log_prefix, retry_count + 1, type(e).__name__, e,
        )
        return await llm_call(
            model_client=model_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cancellation_token=cancellation_token,
            frame=frame,
            parse_check=parse_check,
            parser=parser,
            retry_count=retry_count + 1,
            log_prefix=log_prefix,
            **kwargs,
        )
