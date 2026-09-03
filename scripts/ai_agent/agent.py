from langgraph.prebuilt import create_react_agent
from backend.ai_agent.llm import llm
from backend.ai_agent.memory import memory
from backend.ai_agent.prompt import SYSTEM_PROMPT
from backend.ai_agent.tools import AGENT_TOOLS

agent = create_react_agent(
    model=llm,
    tools=AGENT_TOOLS,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)

def get_assistant_reply(user_input: str, thread_id: str | None = None) -> str:
    if not user_input:
        return ""

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            },
            config={"configurable": {"thread_id": thread_id or "default"}},
        )

        if isinstance(response, dict) and response.get("messages"):
            return response["messages"][-1].content

        return "Sorry, I could not generate a response."
    except Exception as exc:
        return f"Sorry, I could not generate a response: {exc}"
