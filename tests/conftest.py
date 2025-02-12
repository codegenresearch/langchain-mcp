# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests
from mcp import ClientSession, ListToolsResult, Tool
from mcp.types import CallToolResult, TextContent

from langchain_mcp import MCPToolkit


@pytest.fixture(scope="class")
def mcptoolkit():
    session_mock = mock.AsyncMock(spec=ClientSession)
    session_mock.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="read_file",
                description=(
                    "Read the complete contents of a file from the file system. Handles various text encodings "
                    "and provides detailed error messages if the file cannot be read. "
                    "Use this tool when you need to examine the contents of a single file. "
                    "Only works within allowed directories."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#",
                },
            )
        ]
    )
    session_mock.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="MIT License\n\nCopyright (c) 2024 Andrew Wason\n")],
        isError=False,
    )
    toolkit = MCPToolkit(session=session_mock)
    return toolkit


@pytest.fixture(scope="class")
async def mcptool(request, mcptoolkit):
    if not mcptoolkit:
        raise ValueError("MCPToolkit is not initialized.")
    tool = (await mcptoolkit.get_tools())[0]
    request.cls.tool = tool
    return tool


@pytest.mark.usefixtures("mcptool")
class TestMCPToolIntegration(ToolsIntegrationTests):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool = None

    @property
    def tool(self):
        if self._tool is None:
            raise RuntimeError("Tool is not initialized. Please initialize the tool before usage.")
        return self._tool

    @property
    def tool_constructor(self):
        return self.tool

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"path": "LICENSE"}

    async def invoke_tool(self, tool_name, arguments):
        if not self.tool:
            raise RuntimeError("Tool is not initialized. Please initialize the tool before usage.")
        return await self.tool.call(tool_name, arguments)


@pytest.mark.usefixtures("mcptool")
class TestMCPToolUnit(ToolsUnitTests):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool = None

    @property
    def tool(self):
        if self._tool is None:
            raise RuntimeError("Tool is not initialized. Please initialize the tool before usage.")
        return self._tool

    @property
    def tool_constructor(self):
        return self.tool

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"path": "LICENSE"}

    async def invoke_tool(self, tool_name, arguments):
        if not self.tool:
            raise RuntimeError("Tool is not initialized. Please initialize the tool before usage.")
        return await self.tool.call(tool_name, arguments)