# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "langchain-mcp",
#     "langchain-groq",
# ]
# ///


import asyncio
import pathlib
import sys
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp import MCPToolkit
from mcp.types import Tool


async def process_tools_and_messages(
    session: ClientSession, model: ChatGroq, prompt: str
) -> str:
    toolkit = MCPToolkit(session=session)
    await toolkit.initialize()
    tools = await toolkit.get_tools()
    tools_map: Dict[str, Tool] = {tool.name: tool for tool in tools}
    tools_model = model.bind_tools(tools)

    messages: List[BaseMessage] = [HumanMessage(content=prompt)]
    messages.append(await tools_model.ainvoke(messages))

    for tool_call in messages[-1].tool_calls:
        selected_tool = tools_map.get(tool_call["name"].lower())
        if selected_tool:
            tool_msg = await selected_tool.ainvoke(tool_call)
            messages.append(tool_msg)

    result = await (tools_model | StrOutputParser()).ainvoke(messages)
    return result.content if isinstance(result, AIMessage) else str(result)


async def main(prompt: str) -> str:
    model = ChatGroq(model="llama-3.1-8b-instant", stop_sequences=None)  # requires GROQ_API_KEY
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            result = await process_tools_and_messages(session, model, prompt)
            return result


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    result = asyncio.run(main(prompt))
    print(result)