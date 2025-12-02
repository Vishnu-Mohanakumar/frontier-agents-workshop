# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

import httpx
from a2a.client import A2ACardResolver
from agent_framework.a2a import A2AAgent
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

"""
Travel Planning Agent - Scenario 3

This agent demonstrates calling another agent (weather agent) via A2A protocol.
It plans trips based on weather conditions by calling the weather agent as a tool.

Flow:
1. User asks for a 5-day trip plan
2. Travel agent checks multiple European cities with weather agent via A2A
3. Travel agent selects only cities with good weather
4. Returns a planned itinerary
"""


def _create_openai_client() -> OpenAIChatClient:
    """Create an OpenAIChatClient using basic-agent.py pattern."""

    token: str
    endpoint: str
    model_name: str

    if os.environ.get("GITHUB_TOKEN") is not None:
        token = os.environ["GITHUB_TOKEN"]
        endpoint = "https://models.github.ai/inference"
        model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME") or "openai/gpt-4o-mini"
    elif os.environ.get("AZURE_OPENAI_API_KEY") is not None:
        token = os.environ["AZURE_OPENAI_API_KEY"]
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME")
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


async def main():
    """Travel planning agent that calls weather agent via A2A."""
    
    # Get weather agent endpoint from environment
    weather_agent_host = os.getenv("WEATHER_AGENT_URL", "http://localhost:9998")
    
    print("=" * 70)
    print("Travel Planning Agent (Scenario 3)")
    print("=" * 70)
    print(f"\nConnecting to Weather Agent at: {weather_agent_host}\n")

    try:
        # Initialize A2ACardResolver to discover weather agent
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            resolver = A2ACardResolver(httpx_client=http_client, base_url=weather_agent_host)
            
            # Get weather agent card
            print("Discovering weather agent capabilities...")
            agent_card = await resolver.get_agent_card()
            print(f"Found agent: {agent_card.name}")
            print(f"Description: {agent_card.description}\n")

            # Create A2A agent instance for weather
            weather_agent = A2AAgent(
                name=agent_card.name,
                description=agent_card.description,
                agent_card=agent_card,
                url=weather_agent_host,
            )

            # Create local travel planner agent
            travel_planner = _create_openai_client()

            # Define the travel planning flow
            print("Commands:")
            print("  - 'Plan a trip to Paris, Rome, Barcelona'")
            print("  - 'Where should I go in Europe for a sunny vacation?'")
            print("  - 'Plan a 5-day trip with good weather'")
            print("  - ':q' or 'quit' to exit\n")

            while True:
                user_query = input("You: ").strip()

                if not user_query:
                    print("Please enter a query.")
                    continue

                if user_query.lower() in (":q", "quit"):
                    print("\nGoodbye!")
                    break

                print("\n[Travel Agent] Processing your request...\n")

                try:
                    # Step 1: Travel agent reasons about cities to check
                    reasoning_prompt = f"""You are a travel planning expert. 
The user wants: {user_query}

First, decide which 3-4 European cities you want to check for weather.
For each city, you'll call the weather agent to check conditions.

Cities to consider: Paris, Rome, Barcelona, Athens, Lisbon, London, Amsterdam, Berlin

Start by listing which cities you'll check, then we'll get their weather."""

                    print("[Travel Agent] Reasoning about destinations...")
                    reasoning = await travel_planner.get_response(reasoning_prompt)
                    print(f"[Travel Agent] Proposed checks:\n{reasoning}\n")

                    # Step 2: Call weather agent for each city
                    print("[Travel Agent] Checking weather with Weather Agent via A2A...")
                    cities = ["Paris", "Rome", "Barcelona", "Athens", "Lisbon"]
                    
                    weather_results = {}
                    for city in cities:
                        weather_query = f"Is the weather good in {city} for travel?"
                        response = await weather_agent.run(weather_query)
                        weather_text = "\n".join([msg.text for msg in response.messages])
                        weather_results[city] = weather_text
                        print(f"  → {city}: {weather_text[:60]}...")

                    # Step 3: Travel agent plans itinerary based on weather
                    print("\n[Travel Agent] Planning itinerary based on weather...\n")
                    
                    weather_info = "\n".join([f"- {city}: {result}" for city, result in weather_results.items()])
                    
                    planning_prompt = f"""Based on the user's request: {user_query}

And the weather information:
{weather_info}

Plan a complete 5-day itinerary. Include:
1. Which cities you selected and why
2. Day-by-day activities
3. Why you chose this itinerary based on weather
4. Alternative suggestions if needed"""

                    itinerary = await travel_planner.get_response(planning_prompt)
                    print("[Travel Agent] Final Itinerary:")
                    print(f"\n{itinerary}\n")

                except Exception as e:
                    print(f"Error: {e}")
                    print("Make sure the weather agent is running at:")
                    print(f"  python samples/a2a_communication/weather-agent-server.py")

    except Exception as e:
        print(f"Failed to connect to weather agent: {e}")
        print(f"\nTo run this scenario:")
        print(f"1. Terminal 1: python samples/a2a_communication/weather-agent-server.py")
        print(f"2. Terminal 2: python samples/a2a_communication/travel-planning-agent.py")


if __name__ == "__main__":
    asyncio.run(main())
