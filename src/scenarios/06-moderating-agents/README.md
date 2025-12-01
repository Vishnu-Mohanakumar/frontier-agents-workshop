### Scenario 6 - moderating a discussion between agents

Goal: Moderating a complex scenario where multiple agents decide what to do in which order is a core skill when building agentic systems. In this scenario you will design a travel-planning system composed of several specialized agents (for places, weather, activities, and bookings) and a moderator/orchestrator that coordinates them. You will learn how to express high-level constraints such as budget, preferred locations, and diversity of activities, and make sure the agents respect those constraints. This is relevant because real-world agent systems must reliably complete complex, multi-step tasks while staying within user-defined boundaries. By the end, you should understand how a generalist orchestrator like Magentic One can structure and supervise collaboration between many specialized agents.

Task:
- Design a travel agent system where separate agents handle place recommendations, weather validation, activity suggestions, flight booking, and hotel booking.
- Define bounding rules such as total maximum price, preferred or excluded locations, and desired diversity of activities.
- Use a moderator/orchestrator (for example, Magentic One) to coordinate which agent should act next based on current state and constraints.
- Implement logic so that any agent proposal that breaks the rules (e.g., over budget, wrong region) is revised or rejected by the moderator.
- Run end-to-end travel planning conversations and inspect how responsibilities are delegated across agents.

Relevant references
- Magentic One article: https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/

Relevant samples:
- [`samples/magentic`](../../../samples/magentic) – examples of moderating or orchestrating agents.

Input queries:
- "Plan a 7-day city and nature trip in Europe under 2,000 EUR."
- "I want warm weather, good food, and at least three different types of activities."
- "If flights become too expensive, suggest a cheaper alternative itinerary."
- "Explain how each agent contributed to this travel plan."

Tooling tips:
Start by exploring the `samples/magentic` code to see how moderation or orchestration between agents is modeled. Identify where agents are defined, how they are registered with the orchestrator, and how global rules or constraints are passed into the system. When iterating, turn on structured logging or tracing so you can track which agent acted at each step and why proposals were accepted or rejected. Experiment with changing constraints (budget, locations, activity mix) to observe how the moderator adapts the sequence of agent calls.