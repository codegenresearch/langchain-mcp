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

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp import MCPToolkit
from mcp.types import Tool


async def run(tools: list[BaseTool], prompt: str) -> str:
    model = ChatGroq(model="llama-3.1-8b-instant", stop_sequences=None)  # requires GROQ_API_KEY
    tools_map = {tool.name: tool for tool in tools}
    tools_model = model.bind_tools(tools)

    messages: list[BaseMessage] = [HumanMessage(prompt)]
    ai_message = t.cast(AIMessage, await tools_model.ainvoke(messages))
    messages.append(ai_message)

    for tool_call in ai_message.tool_calls:
        selected_tool = tools_map.get(tool_call["name"].lower())
        if selected_tool:
            tool_msg = await selected_tool.ainvoke(tool_call)
            messages.append(tool_msg)

    response = await (tools_model | StrOutputParser()).ainvoke(messages)
    return response.content


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


### Changes Made:
1. **Imports Order and Specificity**: Grouped and ordered imports logically. Imported `BaseTool` from `langchain_core.tools`.
2. **Type Annotations**: Used `list[BaseTool]` for the `tools` parameter in the `run` function.
3. **Casting AIMessage**: Cast the result of `tools_model.ainvoke(messages)` to `AIMessage` using `t.cast`.
4. **Appending AIMessage**: Appended `ai_message` to the `messages` list immediately after invoking the model.
5. **Tool Retrieval**: Directly passed the result of `toolkit.get_tools()` to the `run` function.
6. **Error Handling**: No specific error handling was added in this snippet, but the code is structured to handle missing tools gracefully.
7. **Code Formatting**: Ensured consistent spacing and line breaks for better readability.