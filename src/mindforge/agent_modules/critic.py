import json
import logging
from agent_modules.llm_call import llm_call
from agent_modules.util import CriticResponse, create_model_client
from autogen_core.models import ChatCompletionClient, UserMessage, SystemMessage

with open("prompts/critic_prompt.txt", "r") as f:
    critic_prompt = f.read()
with open("prompts/critic_info.txt", "r") as f:
    critic_info = f.read()


class Critic:
    def __init__(self, critic_model_client=None, override_critic_prompt=None) -> None:
        self.critic_model_client = (
            critic_model_client
            if critic_model_client
            else create_model_client(resonse_format=CriticResponse)
        )
        self.critic_prompt = (
            override_critic_prompt
            if override_critic_prompt
            else eval(f"f'''{critic_prompt}'''")
        )

    async def check_task_success(
        self,
        frame,
        last_action,
        cancellation_token,
        task,
        context,
        communication=None,
        error=None,
        retry_count=0,
        do_conversation=True,
        picked_object=None,
    ):
        # if do_conversation:
        #     return self.check_task_success_conversation(frame, cancellation_token, task, error, retry_count)

        def parse_check(content):
            assert content["success"] in [True, False]
            if "critique" not in content:
                content["critique"] = ""
            return content

        response = await llm_call(
            self.critic_model_client,
            system_prompt=self.critic_prompt,
            user_prompt=critic_info,
            cancellation_token=cancellation_token,
            frame=frame,
            parse_check=parse_check,
            log_prefix="Critic check_task_success: ",
            last_action=last_action,
            task=task,
            error=error,
            context=context,
            communication=communication,
            picked_object=picked_object,
        )
        return response["success"], response["critique"]

    def check_task_success_conversation(self):
        pass
