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
import typing as t
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp import MCPToolkit


async def run(tools: list[BaseTool], prompt: str) -> str:
    tools_map = {tool.name.lower(): tool for tool in tools}
    model = ChatGroq(model="llama-3.1-8b-instant", stop_sequences=None)  # requires GROQ_API_KEY
    tools_model = model.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(prompt)]
    messages.append(await tools_model.ainvoke(messages))
    for tool_call in messages[-1].tool_calls:
        selected_tool = tools_map.get(tool_call["name"].lower())
        if selected_tool is None:
            raise ValueError(f"Tool '{tool_call['name']}' not found.")
        tool_msg = await selected_tool.ainvoke(tool_call)
        messages.append(tool_msg)
    ai_message = t.cast(AIMessage, messages[-1])
    return await (tools_model | StrOutputParser()).ainvoke(messages)


async def main(prompt: str) -> None:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()
            tools = await toolkit.get_tools()
            response = await run(tools, prompt)
            print(response)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    asyncio.run(main(prompt))