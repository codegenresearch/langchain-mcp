# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests
from mcp import ClientSession, ListToolsResult, Tool
from mcp.types import CallToolResult, TextContent

from langchain_mcp import MCPToolkit


@pytest.fixture(scope="class")
def mcptoolkit(request):
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
    yield toolkit
    if issubclass(request.cls, ToolsIntegrationTests):
        session_mock.call_tool.assert_called_with("read_file", arguments={"path": "LICENSE"})


@pytest.fixture(scope="class")
async def mcptool(request, mcptoolkit):
    await mcptoolkit.initialize()  # Initialize the toolkit
    tool = mcptoolkit.get_tools()[0]  # Assuming get_tools is synchronous
    request.cls.tool = tool
    yield tool


This code snippet addresses the feedback by ensuring that the `mcptoolkit` fixture does not call a non-existent `initialize_tools` method and correctly handles the initialization and retrieval of tools. The `get_tools` method is assumed to be synchronous, as per the feedback.