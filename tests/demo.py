# Copyright (C) 2024 Andrew Wason\n# SPDX-License-Identifier: MIT\n\nimport asyncio\nimport pathlib\nimport sys\nfrom typing import List, TypeAlias\n\nfrom langchain_core.messages import HumanMessage, AIMessage, BaseMessage\nfrom langchain_core.output_parsers import StrOutputParser\nfrom langchain_groq import ChatGroq\nfrom mcp import ClientSession, StdioServerParameters\nfrom mcp.client.stdio import stdio_client\nfrom mcp.types import Tool\n\nfrom langchain_mcp import MCPToolkit\nfrom langchain_mcp.tools import BaseTool\n\nToolList: TypeAlias = List[BaseTool]\nMessageList: TypeAlias = List[BaseMessage]\n\n\nasync def initialize_toolkit(session: ClientSession, model: ChatGroq) -> MCPToolkit:\n    toolkit = MCPToolkit(session=session, model=model)\n    await toolkit.initialize()\n    return toolkit\n\n\nasync def process_toolchain(toolkit: MCPToolkit, messages: MessageList) -> str:\n    tools: ToolList = await toolkit.get_tools()\n    tools_map = {tool.name.lower(): tool for tool in tools}\n    tools_model = toolkit.model.bind_tools(tools)\n    messages.append(await tools_model.ainvoke(messages))\n    for tool_call in messages[-1].tool_calls:\n        selected_tool = tools_map[tool_call["name"].lower()]\n        tool_msg = await selected_tool.ainvoke(tool_call)\n        messages.append(tool_msg)\n    final_message = messages[-1]\n    if isinstance(final_message, AIMessage):\n        result = await (toolkit.model | StrOutputParser()).ainvoke(messages)\n        return result\n    else:\n        raise ValueError("Unexpected message type received.")\n\n\nasync def main(prompt: str) -> None:\n    try:\n        model = ChatGroq(model="llama-3.1-8b-instant", stop_sequences=None)  # requires GROQ_API_KEY\n        server_params = StdioServerParameters(\n            command="npx",\n            args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],\n        )\n        async with stdio_client(server_params) as (read, write):\n            async with ClientSession(read, write) as session:\n                toolkit = await initialize_toolkit(session, model)\n                messages: MessageList = [HumanMessage(prompt)]\n                result = await process_toolchain(toolkit, messages)\n                print(result)\n    except ValueError as ve:\n        print(f"Value error: {ve}")\n    except Exception as e:\n        print(f"An unexpected error occurred: {e}")\n\n\nif __name__ == "__main__":\n    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./LICENSE"\n    asyncio.run(main(prompt))\n