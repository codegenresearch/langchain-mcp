# Copyright (C) 2024 Andrew Wason\n# SPDX-License-Identifier: MIT\n\nimport asyncio\nimport pathlib\nimport sys\n\nfrom langchain_core.messages import HumanMessage\nfrom langchain_core.output_parsers import StrOutputParser\nfrom langchain_groq import ChatGroq\nfrom mcp import ClientSession, StdioServerParameters\nfrom mcp.client.stdio import stdio_client\n\nfrom langchain_mcp import MCPToolkit\n\n\nasync def initialize_toolkit(session: ClientSession) -> MCPToolkit:\n    toolkit = MCPToolkit(session=session)\n    await toolkit.initialize()\n    return toolkit\n\n\nasync def main(prompt: str) -> None:\n    model = ChatGroq(model="llama-3.1-8b-instant")  # requires GROQ_API_KEY\n    server_params = StdioServerParameters(\n        command="npx",\n        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],\n    )\n    async with stdio_client(server_params) as (read, write):\n        async with ClientSession(read, write) as session:\n            toolkit = await initialize_toolkit(session)\n            tools = await toolkit.get_tools()\n            tools_map = {tool.name: tool for tool in tools}\n            tools_model = model.bind_tools(tools)\n            messages = [HumanMessage(prompt)]\n            messages.append(await tools_model.ainvoke(messages))\n            for tool_call in messages[-1].tool_calls:\n                selected_tool = tools_map.get(tool_call["name"])\n                if selected_tool is None:\n                    raise ValueError(f"Tool {tool_call[\"name\"]} not found in available tools.")\n                tool_msg = await selected_tool.ainvoke(tool_call)\n                messages.append(tool_msg)\n            result = await (tools_model | StrOutputParser()).ainvoke(messages)\n            print(result)\n\n\nif __name__ == "__main__":\n    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"\n    asyncio.run(main(prompt))\n