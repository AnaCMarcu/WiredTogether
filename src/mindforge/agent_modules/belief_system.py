import logging
import os
from autogen_core.models import ChatCompletionClient, UserMessage, SystemMessage

from agent_modules.llm_call import llm_call
from agent_modules.util import BeliefResponse, create_model_client

_BELIEF_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts", "belief_system")

with open(os.path.join(_BELIEF_DIR, "perception_beliefs.txt"), "r") as f:
    perception_prompt = f.read()
with open(os.path.join(_BELIEF_DIR, "partner_beliefs.txt"), "r") as f:
    partner_prompt = f.read()
with open(os.path.join(_BELIEF_DIR, "interaction_belief.txt"), "r") as f:
    interaction_prompt = f.read()
with open(os.path.join(_BELIEF_DIR, "update_context.txt"), "r") as f:
    context_prompt = f.read()


class BeliefSystem:
    def __init__(
        self,
        number_of_agents,
        belief_model_client=None,
        override_perception_prompt=None,
        override_partner_prompt=None,
        override_interaction_prompt=None,
        override_context_prompt=None,
    ) -> None:
        self.belief_model_client = (
            belief_model_client
            if belief_model_client
            else create_model_client(resonse_format=BeliefResponse)
        )
        self._perception_prompt = override_perception_prompt or perception_prompt
        self._partner_prompt = override_partner_prompt or partner_prompt
        self._interaction_prompt = override_interaction_prompt or interaction_prompt
        self._context_prompt = override_context_prompt or context_prompt
        self.perception_beliefs = ""
        self.partner_beliefs = {i: "" for i in range(number_of_agents - 1)}
        self.interaction_beliefs = ""
        self.task_beliefs = ""

    async def create_perception_beliefs(
        self,
        frame,
        communications,
        error,
        cancellation_token,
    ):

        response = await llm_call(
            self.belief_model_client,
            system_prompt=None,
            user_prompt=self._perception_prompt,
            cancellation_token=cancellation_token,
            frame=frame,
            parse_check=self.parse_check,
            log_prefix="Belief System create_perception_beliefs: ",
            communications=communications,
            error=error,
        )
        self.perception_beliefs = response.get("beliefs", self.perception_beliefs)
        return self.perception_beliefs

    async def update_partner_beliefs(
        self,
        conversations,
        cancellation_token,
    ):

        # for each partner agent
        for i, previous_partner_belief in self.partner_beliefs.items():
            if not conversations or i >= len(conversations):
                continue
            convo = conversations[i]
            # create prompt
            response = await llm_call(
                self.belief_model_client,
                system_prompt=None,
                user_prompt=self._partner_prompt,
                cancellation_token=cancellation_token,
                parse_check=self.parse_check,
                log_prefix="Belief System update_partner_beliefs: ",
                convo=convo,
                previous_partner_belief=previous_partner_belief,
            )
            self.partner_beliefs[i] = response.get("beliefs", previous_partner_belief)
        return self.partner_beliefs

    async def update_interaction_beliefs(
        self,
        task,
        conversations,
        cancellation_token,
    ):
        response = await llm_call(
            self.belief_model_client,
            system_prompt=None,
            user_prompt=self._interaction_prompt,
            cancellation_token=cancellation_token,
            parse_check=self.parse_check,
            log_prefix="Belief System update_interaction_beliefs: ",
            task=task,
            conversations=conversations,
            previous_interaction_beliefs=self.interaction_beliefs,
        )
        self.interaction_beliefs = response.get("beliefs", self.interaction_beliefs)
        return self.interaction_beliefs

    async def update_task_beliefs(self, task, cancellation_token):
        previous_context = self.task_beliefs
        interaction_beliefs = self.interaction_beliefs
        response = await llm_call(
            self.belief_model_client,
            system_prompt=None,
            user_prompt=self._context_prompt,
            cancellation_token=cancellation_token,
            log_prefix="Belief System update_task_beliefs: ",
            task=task,
            previous_context=previous_context,
            interaction_beliefs=interaction_beliefs,
        )
        self.task_beliefs = response.get("beliefs", self.task_beliefs)
        return self.task_beliefs

    def parse_check(self, content):
        assert "beliefs" in content
        return content
