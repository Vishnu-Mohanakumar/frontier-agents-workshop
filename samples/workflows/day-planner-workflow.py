# Copyright (c) Microsoft. All rights reserved.

"""Agent Workflow - Day Planner with Weather and Activities.

This sample demonstrates:
- Multi-stage workflow orchestration
- Parallel execution of weather and activity agents
- Error handling and graceful degradation
- Logging/tracing of workflow steps

Use case: Plan a user's day based on weather conditions and preferences.
Flow:
  1. Gather user location (error if missing)
  2. Parallel: Weather Agent checks conditions + Activity Planner gets preferences
  3. Synthesizer combines results into a day plan
  4. Handle failures with user-friendly error messages
"""

import os
import logging
import json
from typing import Any, Optional

from agent_framework import (
    AgentExecutorResponse,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    AgentExecutorRequest,
    ChatMessage,
    Role,
)
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup logging for tracing workflow steps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Authentication setup
if os.environ.get("GITHUB_TOKEN") is not None:
    token = os.environ["GITHUB_TOKEN"]
    endpoint = "https://models.github.ai/inference"
    logger.info("Using GitHub Token for authentication")
elif os.environ.get("AZURE_OPENAI_API_KEY") is not None:
    token = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    logger.info("Using Azure OpenAI Token for authentication")
else:
    raise ValueError("No authentication token found in environment variables")

async_openai_client = AsyncOpenAI(
    base_url=endpoint,
    api_key=token
)

completion_model_name = os.environ.get("COMPLETION_DEPLOYMENT_NAME", "gpt-4o")
medium_model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME", "gpt-4o")
small_model_name = os.environ.get("SMALL_DEPLOYMENT_MODEL_NAME", "gpt-4o")

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


# ============================================================================
# Stage 1: Input Gatherer - Validates location and extracts user preferences
# ============================================================================

class UserInput(BaseModel):
    """Structured user input for day planning."""
    location: str = Field(..., description="User's location for weather check")
    preferences: Optional[str] = Field(
        default=None,
        description="Activity preferences (e.g., indoor, outdoor, active, relaxing)"
    )
    date_info: Optional[str] = Field(
        default=None,
        description="Date/time info (e.g., tomorrow, weekend, specific date)"
    )


input_gatherer = small_client.create_agent(
    name="InputGatherer",
    instructions=(
        "You are an input validator and extractor. "
        "Parse the user's request and extract:\n"
        "1. Location (required for weather check)\n"
        "2. Activity preferences (optional: indoor/outdoor, active/relaxing, etc.)\n"
        "3. Date info (optional: tomorrow, weekend, specific date)\n\n"
        "If location is missing, respond with ERROR and ask for it.\n"
        "Return a JSON object with these fields.\n"
        "Focus on being concise and clear about what's missing."
    ),
    response_format=UserInput,
)

logger.info("✓ Input Gatherer agent created")


# ============================================================================
# Stage 2a: Weather Agent - Checks weather conditions
# ============================================================================

class WeatherInfo(BaseModel):
    """Weather information for planning."""
    location: str
    condition: str = Field(..., description="Weather condition (sunny, rainy, cloudy, etc.)")
    temperature: str = Field(..., description="Temperature description (e.g., 72°F, warm, cool)")
    recommendations: str = Field(
        ...,
        description="Weather-based activity recommendations"
    )


weather_agent = medium_client.create_agent(
    name="WeatherAgent",
    instructions=(
        "You are a weather expert. Based on the location provided, "
        "describe the expected weather conditions and provide activity recommendations.\n"
        "Return a JSON object with:\n"
        "- condition: weather type (sunny, rainy, cloudy, snowy, etc.)\n"
        "- temperature: temperature description\n"
        "- recommendations: comma-separated activity suggestions based on weather\n\n"
        "If location is unavailable or invalid, provide ERROR in condition field."
    ),
    response_format=WeatherInfo,
)

logger.info("✓ Weather Agent created")


# ============================================================================
# Stage 2b: Activity Planner - Suggests activities based on preferences
# ============================================================================

class ActivityPlan(BaseModel):
    """Activity plan based on user preferences."""
    activities: str = Field(
        ...,
        description="Comma-separated list of suggested activities"
    )
    schedule: str = Field(
        ...,
        description="Suggested time schedule (morning, afternoon, evening activities)"
    )
    notes: str = Field(
        ...,
        description="Any special notes or tips"
    )


activity_planner = small_client.create_agent(
    name="ActivityPlanner",
    instructions=(
        "You are a creative activity planner. Based on the user's preferences, "
        "suggest a variety of activities for their day.\n"
        "Consider:\n"
        "- Indoor vs outdoor preferences\n"
        "- Active vs relaxing preferences\n"
        "- Time of day optimization\n\n"
        "Return a JSON object with:\n"
        "- activities: 4-6 activity suggestions\n"
        "- schedule: suggested time distribution (morning, afternoon, evening)\n"
        "- notes: any tips or recommendations\n\n"
        "Be creative and practical with suggestions."
    ),
    response_format=ActivityPlan,
)

logger.info("✓ Activity Planner created")


# ============================================================================
# Stage 3: Synthesizer - Combines weather and activities into day plan
# ============================================================================

day_synthesizer = completion_client.create_agent(
    name="DaySynthesizer",
    instructions=(
        "You are an expert day planner. You receive weather information and activity suggestions. "
        "Synthesize them into a cohesive, personalized day plan.\n\n"
        "Your plan should:\n"
        "1. Start with weather context\n"
        "2. List time-based schedule (morning, afternoon, evening)\n"
        "3. Highlight activities that match weather conditions\n"
        "4. Include practical tips (e.g., bring umbrella if rainy)\n"
        "5. Keep it concise but actionable\n\n"
        "Format as a numbered day plan with specific times and activities."
    ),
)

logger.info("✓ Day Synthesizer created")


# ============================================================================
# Error Handling
# ============================================================================

def validate_location(message: Any) -> bool:
    """Check if location was successfully extracted."""
    if not isinstance(message, AgentExecutorResponse):
        logger.warning("Invalid message type for location validation")
        return False
    try:
        user_input = UserInput.model_validate_json(message.agent_run_response.text)
        has_location = user_input.location and user_input.location.upper() != "ERROR"
        if not has_location:
            logger.error("Location validation failed: location is missing or invalid")
        return has_location
    except Exception as e:
        logger.error(f"Error parsing location: {e}")
        return False


def handle_missing_location(context: Any) -> str:
    """Provide user-friendly error message for missing location."""
    error_msg = (
        "❌ Day Planning Failed\n\n"
        "I need your location to check the weather and plan your day.\n"
        "Please provide a location (e.g., 'New York', 'San Francisco') and try again.\n\n"
        "Example: 'Plan my day tomorrow in Boston including outdoor activities.'"
    )
    logger.error("Workflow stopped: missing location")
    return error_msg


logger.info("✓ Error handlers configured")


# ============================================================================
# Combined Results Model
# ============================================================================

class CombinedPlanData(BaseModel):
    """Combined weather and activity data for synthesis."""
    weather: WeatherInfo
    activities: ActivityPlan


# ============================================================================
# Bridge Step - Collects responses and passes to synthesizer
# ============================================================================

join_call_count = 0
weather_result = None
activity_result = None


@executor(id="join_weather_activities")
async def join_parallel_results(message: AgentExecutorResponse, ctx: WorkflowContext) -> None:
    """Join responses from parallel agents using shared state."""
    global join_call_count, weather_result, activity_result
    
    join_call_count += 1
    logger.info(f"⏱ Join step called ({join_call_count}/2): Processing parallel agent response...")
    
    try:
        # Determine which agent this is from based on response content
        response_text = message.agent_run_response.text
        
        if "condition" in response_text.lower() and "temperature" in response_text.lower():
            # This is weather response
            weather_result = WeatherInfo.model_validate_json(response_text)
            logger.info("✓ Weather agent response captured")
        elif "activities" in response_text.lower() and "schedule" in response_text.lower():
            # This is activity response
            activity_result = ActivityPlan.model_validate_json(response_text)
            logger.info("✓ Activity planner response captured")
        
        # When both are ready, forward to synthesizer
        if weather_result and activity_result:
            join_call_count = 0
            combined_context = f"""Weather Information:
- Location: {weather_result.location}
- Condition: {weather_result.condition}
- Temperature: {weather_result.temperature}
- Weather-based recommendations: {weather_result.recommendations}

Activity Preferences & Suggestions:
- Suggested activities: {activity_result.activities}
- Recommended schedule: {activity_result.schedule}
- Notes: {activity_result.notes}

Now synthesize this into a complete day plan."""
            
            logger.info("✓ Join step: Both agents completed, forwarding to synthesizer")
            weather_result = None
            activity_result = None
            
            await ctx.send_message(AgentExecutorRequest(
                messages=[ChatMessage(role=Role.USER, content=combined_context)],
                should_respond=True
            ))
        else:
            logger.info("⏱ Waiting for second agent...")
    except Exception as e:
        logger.error(f"Error in join step: {e}")
        join_call_count = 0
        weather_result = None
        activity_result = None


logger.info("✓ Join step configured")


# ============================================================================
# Build Workflow with Parallel Execution and Join
# ============================================================================

join_executor = join_parallel_results

workflow = (
    WorkflowBuilder()
    .set_start_executor(input_gatherer)
    # Route based on location validation
    .add_edge(input_gatherer, weather_agent, condition=validate_location)
    .add_edge(input_gatherer, activity_planner, condition=validate_location)
    # Both agents converge at join step (waits for both)
    .add_edge(weather_agent, join_executor)
    .add_edge(activity_planner, join_executor)
    # Join routes to synthesizer after both complete
    .add_edge(join_executor, day_synthesizer)
    .build()
)

logger.info("✓ Workflow built with join pattern for parallel stages")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Launch the day planner workflow in DevUI."""
    logger.info("\n" + "="*70)
    logger.info("Starting Day Planner Workflow")
    logger.info("="*70)
    logger.info("Available at: http://localhost:8093")
    logger.info("\nWorkflow Structure:")
    logger.info("  1. Input Gatherer → Extract location & preferences")
    logger.info("  2. Parallel Stage (triggered once):")
    logger.info("     - Weather Agent → Check conditions")
    logger.info("     - Activity Planner → Suggest activities")
    logger.info("  3. Join Step → Waits for both, combines responses")
    logger.info("  4. Day Synthesizer → Creates final day plan")
    logger.info("\nTry queries like:")
    logger.info("  • 'Plan my day tomorrow in New York including outdoor activities'")
    logger.info("  • 'Check the weather for the weekend in Seattle and suggest activities'")
    logger.info("  • 'I want a relaxing day in Miami - what should I do?'")
    logger.info("="*70 + "\n")

    from agent_framework.devui import serve

    serve(entities=[workflow], port=8093, auto_open=True)


if __name__ == "__main__":
    main()
