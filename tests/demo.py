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

from langchain_mcp import MCPToolkit, Tool


class ToolManager:
    def __init__(self, session: ClientSession):
        self.session = session
        self.toolkit = MCPToolkit(session=session)
        self.tools: Optional[List[Tool]] = None
        self.tools_map: Optional[Dict[str, Tool]] = None

    async def initialize_tools(self) -> None:
        if self.tools is None:
            self.tools = await self.toolkit.get_tools()
            self.tools_map = {tool.name.lower(): tool for tool in self.tools}
        else:
            raise RuntimeError("Tools are already initialized.")

    def get_tool(self, tool_name: str) -> Tool:
        if self.tools_map is None:
            raise RuntimeError("Tools are not initialized. Call initialize_tools() first.")
        tool = self.tools_map.get(tool_name.lower())
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return tool


async def run(model: ChatGroq, tool_manager: ToolManager, prompt: str) -> str:
    messages: List[HumanMessage | AIMessage] = [HumanMessage(prompt)]
    messages.append(await model.ainvoke(messages))
    for tool_call in messages[-1].tool_calls:
        selected_tool = tool_manager.get_tool(tool_call["name"])
        tool_msg = await selected_tool.ainvoke(tool_call)
        messages.append(tool_msg)
    result = await (model | StrOutputParser()).ainvoke(messages)
    return result


async def main(prompt: str) -> None:
    model = ChatGroq(model="llama-3.1-8b-instant")  # requires GROQ_API_KEY
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            tool_manager = ToolManager(session=session)
            await tool_manager.initialize_tools()
            result = await run(model, tool_manager, prompt)
            print(result)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    asyncio.run(main(prompt))