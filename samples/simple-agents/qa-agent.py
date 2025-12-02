# Copyright (c) Microsoft. All rights reserved.
import os
import asyncio
from typing import Annotated
from agent_framework import ChatAgent
from agent_framework import ai_function
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

"""
Q&A Agent Example

A simple interactive agent that answers questions about general topics.
Demonstrates basic agent setup with an interactive conversation loop.
"""

# Authentication setup - supports both GitHub and Azure OpenAI
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

model_name = os.environ.get("MEDIUM_DEPLOYMENT_MODEL_NAME")
if not model_name:
    # Fallback to a default model if not specified
    if os.environ.get("GITHUB_TOKEN") is not None:
        model_name = "openai/gpt-4o-mini"
    else:
        # For Azure, you still need to set MEDIUM_DEPLOYMENT_MODEL_NAME
        raise ValueError(
            "MEDIUM_DEPLOYMENT_MODEL_NAME environment variable not set. "
            "For Azure OpenAI, set this to your deployment name. "
            "For GitHub Models, it defaults to 'openai/gpt-4o-mini'."
        )

client = OpenAIChatClient(
    model_id=model_name,
    api_key=token,
    async_client=async_openai_client
)


@ai_function(
    name="search_knowledge_base",
    description="Search for information on a given topic"
)
def search_knowledge_base(topic: Annotated[str, "The topic to search for"]) -> str:
    """
    Search the knowledge base for information about a topic.
    
    Args:
        topic: The topic to search for
        
    Returns:
        str: Information about the topic
    """
    # Simulated knowledge base responses
    knowledge_base = {
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "azure": "Azure is a cloud computing platform by Microsoft offering various services.",
        "agent": "An agent is an AI system that can perceive its environment and take actions to achieve goals.",
        "machine learning": "Machine learning is a subset of AI where systems learn patterns from data without explicit programming.",
    }
    
    topic_lower = topic.lower()
    for key, value in knowledge_base.items():
        if key in topic_lower:
            return value
    
    return f"Information about '{topic}' is not available in the knowledge base."


# Create the Q&A agent
agent = ChatAgent(
    chat_client=client,
    name="QAAgent",
    instructions=(
        "You are a helpful Q&A agent. Answer user questions accurately and concisely. "
        "If you don't know the answer, admit it and suggest alternative ways to find information. "
        "Use the search_knowledge_base tool when appropriate to provide better answers."
    ),
    tools=[search_knowledge_base],
)


async def main():
    """Run the interactive Q&A agent session."""
    thread = agent.get_new_thread()
    print("=== Q&A Agent - Interactive Session ===")
    print("Ask me anything! Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            print("\nAgent: ", end="", flush=True)
            result = await agent.run(user_input, thread=thread)
            print(result.text)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    asyncio.run(main())
