# Copyright (C) 2024 Andrew Wason\n# SPDX-License-Identifier: MIT\n\nfrom unittest import mock\n\nimport pytest\nfrom langchain_tests.integration_tests import ToolsIntegrationTests\nfrom mcp import ClientSession, ListToolsResult, Tool\nfrom mcp.types import CallToolResult, TextContent\n\nfrom langchain_mcp import MCPToolkit\n\n\n@pytest.fixture(scope="class")\ndef mcptoolkit(request):\n    session_mock = mock.AsyncMock(spec=ClientSession)\n    session_mock.list_tools.return_value = ListToolsResult(\n        tools=[\n            Tool(\n                name="read_file",\n                description=(\n                    "Read the complete contents of a file from the file system. Handles various text encodings "\n                    "and provides detailed error messages if the file cannot be read. "\n                    "Use this tool when you need to examine the contents of a single file. "\n                    "Only works within allowed directories."\n                ),\n                inputSchema={\n                    "type": "object",\n                    "properties": {"path": {"type": "string"}},\n                    "required": ["path"],\n                    "additionalProperties": False,\n                    "$schema": "http://json-schema.org/draft-07/schema#",\n                },\n            )\n        ]\n    )\n    session_mock.call_tool.return_value = CallToolResult(\n        content=[TextContent(type="text", text="MIT License\n\nCopyright (c) 2024 Andrew Wason\n")],\n        isError=False,\n    )\n    toolkit = MCPToolkit(session=session_mock)\n    toolkit.initialize()  # Ensure the toolkit is initialized\n    tools = toolkit.get_tools()\n    if not tools:\n        raise RuntimeError("No tools initialized in the toolkit.")\n    request.cls.toolkit = toolkit\n    yield toolkit\n    if issubclass(request.cls, ToolsIntegrationTests):\n        session_mock.call_tool.assert_called_with("read_file", arguments={"path": "LICENSE"})\n\n\n@pytest.fixture(scope="class")\nasync def mcptool(request, mcptoolkit):\n    tools = mcptoolkit.get_tools()\n    if not tools:\n        raise RuntimeError("Tools are not initialized.")\n    tool = tools[0]\n    request.cls.tool = tool\n    yield tool\n