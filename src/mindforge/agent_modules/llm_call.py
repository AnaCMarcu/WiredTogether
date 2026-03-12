import json
from agent_modules.util import load_json
import re
import logging
import time
from autogen_core.models import ChatCompletionClient, UserMessage, SystemMessage

def _safe_format(template: str, **kwargs) -> str:
    """Format a template string, replacing missing keys with 'N/A' instead of crashing.
 
    This prevents the raw template (with literal {reward_text} etc.) from being
    sent to the LLM when a placeholder is not provided.
    """
    # Find all {placeholder} patterns (not {{escaped}})
    placeholders = set(re.findall(r'(?<!\{)\{(\w+)\}(?!\})', template))
    # Fill any missing keys with a default
    for key in placeholders:
        if key not in kwargs:
            kwargs[key] = "N/A"
            logging.warning(f"Prompt placeholder '{{{key}}}' not provided, using 'N/A'")
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError) as e:
        logging.error(f"Template formatting failed even after defaults: {e}")
        return template


async def llm_call(
    model_client: ChatCompletionClient,
    user_prompt,
    cancellation_token,
    system_prompt=None,
    frame=None,
    parse_check=None,
    retry_count=0,
    log_prefix="LLM call",
    pred_prompt=None,
    pred_frame=None,
    **kwargs,
):
    # if failed more than 5 times, return empty fallback instead of crashing
    if retry_count > 5:
        logging.error(f"{log_prefix} Too many retries (6), returning empty response")
        return {}

    # Build prompt
    try:
        filled_user_prompt = user_prompt.format(**kwargs)
    except KeyError as e:
        logging.error(f"{log_prefix} Missing prompt placeholder {e}, using raw prompt")
        filled_user_prompt = user_prompt
    if frame is None:
        user_message = UserMessage(content=[filled_user_prompt], source="user")
    else:
        user_message = UserMessage(content=[filled_user_prompt, frame], source="user")

    if system_prompt is not None:
        full_prompt = [SystemMessage(content=system_prompt), user_message]
    else:
        full_prompt = [user_message]

    if pred_prompt is not None and pred_frame is not None:
        print("using pred")
        prediction_message = UserMessage(
            content=[pred_prompt, pred_frame], source="user"
        )
        full_prompt.append(prediction_message)

    logging.info(
        f"{log_prefix} System prompt: {system_prompt}, user prompt: {filled_user_prompt}"
    )

    # call model
    try:
        response = await model_client.create(
            full_prompt, cancellation_token=cancellation_token
        )
    except Exception as e:
        error_str = str(e)
        logging.error(f"{log_prefix} Error calling LLM (attempt {retry_count + 1}): {error_str[:300]}")
        time.sleep(1)
        return await llm_call(
            model_client=model_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cancellation_token=cancellation_token,
            frame=frame,
            retry_count=retry_count + 1,
            log_prefix=log_prefix,
            pred_prompt=pred_prompt,
            pred_frame=pred_frame,
            **kwargs,
            )

    logging.info(f"{log_prefix} Response: {response.content}")

    # parse response
    try:
        content = load_json(response.content)
        if parse_check is not None:
            content = parse_check(content)
        return content
    except (json.JSONDecodeError, AssertionError, KeyError, Exception) as e:
        logging.error(
            f"{log_prefix} Error parsing response (attempt {retry_count + 1}): {type(e).__name__}: {e}"
        )
        return await llm_call(
            model_client=model_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cancellation_token=cancellation_token,
            frame=frame,
            retry_count=retry_count + 1,
            log_prefix=log_prefix,
            pred_prompt=pred_prompt,
            pred_frame=pred_frame,
            **kwargs,
        )
