import json
from typing import Dict, List, Optional, Any, Union

from ai_agent.llm import chat, is_configured
from ai_agent.memory import memory
from ai_agent.prompt import SYSTEM_PROMPT
from ai_agent.tools import (
    AGENT_TOOLS,
    coldest_panchayat,
    hottest_panchayat,
    highest_irrigation_demand_panchayat,
    list_panchayats,
    get_panchayat,
    extract_location_from_query,
    lookup_external_region_weather,
)


def _dispatch_tools(user_input: str, telemetry: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inspects query intent and executes relevant data tools against current telemetry."""
    if not telemetry:
        return []

    tools_executed = []
    q_lower = user_input.lower()

    # Tool 1: Coldest Panchayat / Frost / Inversion
    if any(k in q_lower for k in ["coldest", "cold", "frost", "lowest temp", "chilling", "drainage", "inversion"]):
        res = coldest_panchayat(telemetry)
        if res:
            tools_executed.append({
                "tool": "coldest_panchayat",
                "result": res
            })

    # Tool 2: Hottest Panchayat / Heat stress
    if any(k in q_lower for k in ["hottest", "warmest", "highest temp", "heat stress", "thermal"]):
        res = hottest_panchayat(telemetry)
        if res:
            tools_executed.append({
                "tool": "hottest_panchayat",
                "result": res
            })

    # Tool 3: Highest Irrigation Demand / Water
    if any(k in q_lower for k in ["irrigation", "water demand", "liters", "et0", "evapotranspiration", "watering"]):
        res = highest_irrigation_demand_panchayat(telemetry)
        if res:
            tools_executed.append({
                "tool": "highest_irrigation_demand_panchayat",
                "result": res
            })

    # Tool 4: List all panchayats
    if any(k in q_lower for k in ["list panchayat", "all panchayat", "which panchayat", "zones", "villages"]):
        res = list_panchayats(telemetry)
        if res:
            tools_executed.append({
                "tool": "list_panchayats",
                "result": res
            })

    # Tool 5: Specific Panchayat lookup in active region
    for p in telemetry.get("panchayats", []):
        p_name = p.get("panchayat_name", "")
        if p_name and p_name.lower() in q_lower:
            tools_executed.append({
                "tool": "get_panchayat",
                "panchayat": p_name,
                "result": p
            })
            break

    # Tool 6: External Village / Region Lookup (e.g. Pune, Nashik, Darjeeling, etc.)
    ext_loc = extract_location_from_query(user_input)
    current_reg = telemetry.get("region_name", "").lower()
    if ext_loc and ext_loc.lower() not in current_reg:
        ext_wx = lookup_external_region_weather(ext_loc)
        if ext_wx:
            tools_executed.append({
                "tool": "lookup_external_region_weather",
                "location": ext_loc,
                "result": ext_wx
            })

    return tools_executed


def get_assistant_reply(
    user_input: str,
    telemetry: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = "default",
    return_dict: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Main entrypoint for AI Agent queries.
    1. Dispatches data inspection tools based on user intent.
    2. Builds grounded context with live 1km telemetry and tool findings.
    3. Retrieves multi-turn history from memory and calls LLM.
    4. Updates memory and returns the response.
    """
    if not user_input or not user_input.strip():
        empty_res = "Please provide a question or instruction for the AI Agent."
        return {"reply": empty_res, "thread_id": thread_id or "default", "tools_used": []} if return_dict else empty_res

    t_id = thread_id or "default"
    safe_telemetry = telemetry or {}

    # Run data tools
    tool_runs = _dispatch_tools(user_input, safe_telemetry)
    tools_used_names = [t["tool"] for t in tool_runs]

    # Build context prompt
    context_data = {
        "region_name": safe_telemetry.get("region_name", "Target Region"),
        "timestamp": safe_telemetry.get("timestamp_label", safe_telemetry.get("live_meta", {}).get("live_time", "Current")),
        "metrics": safe_telemetry.get("metrics", {})
    }

    prompt_sections = [
        f"{SYSTEM_PROMPT}\n",
        f"### DOWN-SCALED 1KM TELEMETRY CONTEXT:\n```json\n{json.dumps(context_data, indent=2)}\n```\n"
    ]

    if tool_runs:
        prompt_sections.append(
            f"### REAL-TIME TOOL INSPECTION RESULTS:\n```json\n{json.dumps(tool_runs, indent=2)}\n```\n"
        )

    # Retrieve prior conversation messages
    history = memory.get_messages(t_id)

    # Prepare message payload for chat completion
    messages_payload: List[Dict[str, str]] = [
        {"role": "system", "content": "\n".join(prompt_sections)}
    ]

    for h in history[-6:]:  # Keep recent history
        messages_payload.append(h)

    # Add current user query
    messages_payload.append({"role": "user", "content": user_input})

    # Call LLM / Expert Engine
    reply = chat(messages_payload, telemetry=safe_telemetry)

    # Update conversation memory
    memory.add_message("user", user_input, thread_id=t_id)
    memory.add_message("assistant", reply, thread_id=t_id)

    if return_dict:
        return {
            "reply": reply,
            "thread_id": t_id,
            "tools_used": tools_used_names
        }

    return reply

