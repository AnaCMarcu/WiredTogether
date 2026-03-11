import json
from agent_modules.util import load_json
import logging
import time
from autogen_core.models import ChatCompletionClient, UserMessage, SystemMessage


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
    # if failed more that 5 times, raise error
    if retry_count > 5:
        raise ValueError(log_prefix, " Too many retries: ")

    # Build prompt
    filled_user_prompt = user_prompt.format(**kwargs)
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
        # print(f"full_prompt: {full_prompt}")
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
        # # if failed, try again and increment retry counter
        # logging.error(log_prefix, "Error Calling LLM: ", e)
        # time.sleep(1)
        # return await llm_call(
        #     model_client=model_client,
        #     system_prompt=system_prompt,
        #     user_prompt=user_prompt,
        #     cancellation_token=cancellation_token,
        #     frame=frame,
        #     retry_count=retry_count + 1,
        #     log_prefix=log_prefix,
        #     pred_prompt=pred_prompt,
        #     pred_frame=pred_frame,
        #     **kwargs,
        # )

    logging.info(f"{log_prefix} Response: {response.content}")

    # parse response
    try:
        content = load_json(response.content)
        if parse_check is not None:
            content = parse_check(content)
        return content
    except json.JSONDecodeError as e:
        logging.error(
            log_prefix,
            f"Error decoding JSON: {e} \n Given string is: {response.content} \n",
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
