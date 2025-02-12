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

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp import MCPToolkit


class ToolIntegration:
    def __init__(self, session: ClientSession):
        self.session = session
        self.toolkit = MCPToolkit(session=session)
        self.tools = []
        self.tools_map = {}
        self.tools_model = None
        self._initialized = False

    async def initialize(self):
        await self.toolkit.initialize()
        self.tools = await self.toolkit.get_tools()
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.tools_model = self.model.bind_tools(self.tools)
        self._initialized = True

    @property
    def model(self):
        return ChatGroq(model="llama-3.1-8b-instant")  # requires GROQ_API_KEY

    async def run_tool(self, prompt: str) -> str:
        if not self._initialized:
            raise RuntimeError("Toolkit is not initialized. Call initialize() first.")

        messages = [HumanMessage(prompt)]
        messages.append(await self.tools_model.ainvoke(messages))
        for tool_call in messages[-1].tool_calls:
            selected_tool = self.tools_map[tool_call["name"].lower()]
            tool_msg = await selected_tool.ainvoke(tool_call)
            messages.append(tool_msg)
        result = await (self.tools_model | StrOutputParser()).ainvoke(messages)
        return result


async def main(prompt: str) -> None:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            tool_integration = ToolIntegration(session)
            await tool_integration.initialize()
            result = await tool_integration.run_tool(prompt)
            print(result)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"
    asyncio.run(main(prompt))