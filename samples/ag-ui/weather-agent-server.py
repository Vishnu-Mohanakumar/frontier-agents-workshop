# Copyright (c) Microsoft. All rights reserved.
import os
import asyncio
import json
from random import randint
from typing import Annotated
from pathlib import Path

from agent_framework.openai import OpenAIChatClient
from pydantic import Field

from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

# Persistent storage file
USER_DATA_FILE = Path.home() / ".weather_agent" / "user_data.json"


"""
Weather Agent Server using basic-agent.py pattern

Demonstrates OpenAIChatClient usage for weather-related queries.
Shows function calling capabilities with weather tools.
"""


if os.environ.get("GITHUB_TOKEN") is not None:
    token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.github.ai/inference"
    print("Using GitHub Token for authentication")
elif os.environ.get("AZURE_OPENAI_API_KEY") is not None:
    token = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    print("Using Azure OpenAI Token for authentication")
else:
    raise ValueError("Neither GITHUB_TOKEN nor AZURE_OPENAI_API_KEY is set")

async_openai_client = AsyncOpenAI(
    base_url=endpoint,
    api_key=token
)

completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")
small_model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")

completion_client = OpenAIChatClient(
    model_id=completion_model_name,
    api_key=token,
    async_client=async_openai_client
)

medium_client = OpenAIChatClient(
    model_id=medium_model_name,
    api_key=token,
    async_client=async_openai_client
)

small_client = OpenAIChatClient(
    model_id=small_model_name,
    api_key=token,
    async_client=async_openai_client
)


def load_user_data():
    """Load user data from persistent storage."""
    try:
        if USER_DATA_FILE.exists():
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load user data: {e}")
    return {"location": "Seattle"}


def save_user_data(data):
    """Save user data to persistent storage."""
    try:
        USER_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save user data: {e}")


def get_weather_at_location(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the realtime weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy", "partly cloudy"]
    temp = randint(40, 85)
    condition = conditions[randint(0, len(conditions) - 1)]
    return f"The weather in {location} is {condition} with a high of {temp}°F."


async def get_weather_for_user(message: str) -> str:
    """Get weather information from the agent."""
    client = small_client
    print(f"User: {message}")
    response = await client.get_response(message, tools=get_weather_at_location)
    return response


async def main() -> None:
    print("=" * 70)
    print("Weather Agent Server (basic-agent pattern)")
    print("=" * 70)
    
    # Load user data from storage
    user_data = load_user_data()
    current_location = user_data.get("location", "Seattle")
    print(f"\nLoaded location from memory: {current_location}")
    print("\nWeather Agent Ready - Awaiting queries")
    print("(Your location is remembered between sessions)\n")
    
    while True:
        message = input("You: ").strip()
        
        if not message:
            print("Message cannot be empty.")
            continue
        
        # Handle location change commands
        if message.lower().startswith("my location is "):
            current_location = message.replace("my location is ", "", 1).strip()
            user_data["location"] = current_location
            save_user_data(user_data)
            print(f"Assistant: Location set to {current_location}\n")
            continue
            
        if message.lower() in (":q", "quit"):
            print("\nGoodbye!")
            break
        
        # Always include current location in the message for context
        if "where i am" in message.lower() or ("weather" in message.lower() and current_location not in message.lower()):
            # If asking about "where I am" or just "weather", use saved location
            full_message = f"What's the weather in {current_location}?"
        else:
            full_message = message
        
        try:
            response = await get_weather_for_user(full_message)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
