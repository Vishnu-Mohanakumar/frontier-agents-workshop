# Copyright (c) Microsoft. All rights reserved.

import os
from random import randint
from typing import Annotated, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import Field


load_dotenv()


def is_weather_good(
    location: Annotated[str, Field(description="The location to check weather for.")],
    date: Annotated[str, Field(description="The date to check (e.g., 'Dec 25, 2025')")] = None,
) -> str:
    """Check if weather is good (sunny) at a location on a given date."""
    # Simulated weather data: some locations have good weather
    good_weather_locations = ["Paris", "Barcelona", "Rome", "Athens", "Lisbon"]
    
    is_good = location in good_weather_locations
    conditions = ["sunny" if is_good else "rainy", "cloudy", "stormy"]
    condition = conditions[randint(0, min(1 if is_good else 2, len(conditions)-1))]
    
    temp = randint(15, 25) if is_good else randint(5, 15)
    date_str = f" on {date}" if date else ""
    
    return f"Weather in {location}{date_str}: {condition}, {temp}°C. {'Good for travel!' if is_good else 'Not ideal for travel.'}"


def _create_openai_client() -> OpenAIChatClient:
    """Create an OpenAIChatClient using basic-agent.py pattern."""

    token: str
    endpoint: str
    model_name: str

    if os.environ.get("GITHUB_TOKEN") is not None:
        token = os.environ["GITHUB_TOKEN"]
        endpoint = "https://models.github.ai/inference"
        model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME") or "openai/gpt-4o-mini"
    elif os.environ.get("AZURE_OPENAI_API_KEY") is not None:
        token = os.environ["AZURE_OPENAI_API_KEY"]
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
    else:
        raise RuntimeError(
            "No OpenAI credentials found. Set GITHUB_TOKEN or AZURE_OPENAI_API_KEY."
        )

    async_openai_client = AsyncOpenAI(
        base_url=endpoint,
        api_key=token,
    )

    return OpenAIChatClient(
        model_id=model_name,
        api_key=token,
        async_client=async_openai_client,
    )


class WeatherAgentExecutor(AgentExecutor):
    """Weather agent that answers questions about weather conditions.
    
    Reuses the basic-agent.py pattern from samples/simple-agents/basic-agent.py
    but exposes it via A2A protocol for other agents to use.
    """

    def __init__(self):
        self.agent = _create_openai_client()

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task

        if not context.message:
            raise Exception('No message provided')

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        # Enhance the query with weather expertise context
        query = f"You are a weather expert. Answer questions about weather conditions. {context.get_user_input()}"

        # Use the agent framework chat client with weather tool
        # Following the basic-agent.py pattern
        response = await self.agent.get_response(query, tools=is_weather_good)

        # Coerce response to string for A2A
        response_text = str(response)

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                append=False,
                context_id=task.context_id,
                task_id=task.id,
                artifact=new_text_artifact(response_text),
            ),
        )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=task.context_id,
                task_id=task.id,
                status=TaskStatus.SUCCEEDED,
            ),
        )
