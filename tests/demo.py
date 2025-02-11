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
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp import MCPToolkit


async def initialize_tools(session: ClientSession) -> Dict[str, 'Tool']:
    toolkit = MCPToolkit(session=session)
    tools = await toolkit.get_tools()
    return {tool.name: tool for tool in tools}


async def run(prompt: str, tools: Dict[str, 'Tool'], model: ChatGroq) -> str:
    messages: List[HumanMessage | AIMessage] = [HumanMessage(prompt)]
    tools_model = model.bind_tools(list(tools.values()))
    messages.append(await tools_model.ainvoke(messages))
    
    for tool_call in messages[-1].tool_calls:
        selected_tool = tools.get(tool_call["name"].lower())
        if selected_tool:
            tool_msg = await selected_tool.ainvoke(tool_call)
            messages.append(tool_msg)
        else:
            raise ValueError(f"Tool {tool_call['name']} not found.")
    
    result = await (tools_model | StrOutputParser()).ainvoke(messages)
    return result


async def main(prompt: str) -> None:
    model = ChatGroq(model="llama-3.1-8b-instant", stop_sequences=None)  # requires GROQ_API_KEY
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            tools = await initialize_tools(session)
            result = await run(prompt, tools, model)
            print(result)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    asyncio.run(main(prompt))