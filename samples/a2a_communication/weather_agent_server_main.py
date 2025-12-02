# Copyright (c) Microsoft. All rights reserved.

import click
import uvicorn

from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    GetTaskRequest,
    GetTaskResponse,
    SendMessageRequest,
    SendMessageResponse,
)

from samples.a2a_communication.server.agent_executor import HelloWorldAgentExecutor


class A2AWeatherRequestHandler(DefaultRequestHandler):
    """A2A Request Handler for the Weather Agent."""

    def __init__(
        self, agent_executor: AgentExecutor, task_store: InMemoryTaskStore
    ):
        super().__init__(agent_executor, task_store)

    async def on_get_task(
        self, request: GetTaskRequest, *args, **kwargs
    ) -> GetTaskResponse:
        return await super().on_get_task(request, *args, **kwargs)

    async def on_message_send(
        self, request: SendMessageRequest, *args, **kwargs
    ) -> SendMessageResponse:
        return await super().on_message_send(request, *args, **kwargs)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=9998)
def main(host: str, port: int):
    """Start the weather agent server (Scenario 3).
    
    This agent can be called by travel planning agents via A2A protocol
    to check weather conditions for potential travel destinations.
    
    Reuses HelloWorldAgentExecutor from server/agent_executor.py
    """
    skill = AgentSkill(
        id='check_weather_conditions',
        name='Check weather conditions',
        description=(
            'The agent can check weather conditions at specific locations '
            'and determine if they are suitable for travel.'
        ),
        tags=['weather', 'travel', 'conditions'],
        examples=[
            'Is the weather good in Paris in December?',
            'What is the weather like in Barcelona?',
            'Check weather for Rome and Athens in January.',
        ],
    )

    agent_card = AgentCard(
        name='Weather Agent (A2A)',
        description=(
            'A weather specialist agent that checks conditions at locations '
            'and recommends suitable travel destinations based on weather. '
            'This agent is exposed via A2A protocol for other agents to use.'
        ),
        url=f'http://{host}:{port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(
            input_modes=['text'],
            output_modes=['text'],
            streaming=False,
        ),
        skills=[skill],
        examples=[
            'Is the weather good in Paris in December?',
            'What is the weather like in Barcelona?',
            'Check weather for Rome and Athens in January.',
        ],
    )

    task_store = InMemoryTaskStore()
    request_handler = A2AWeatherRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=task_store,
    )

    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    
    print(f"Weather Agent (A2A) starting on http://{host}:{port}")
    print(f"This agent can be called by other agents via A2A protocol")
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    main()
