# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import warnings
from collections.abc import Callable

import pydantic
import pydantic_core
import typing_extensions as t
from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException
from mcp import ClientSession


class MCPToolkit(BaseToolkit):
    """\n    MCP server toolkit\n    """

    session: ClientSession
    """The MCP session used to obtain the tools"""

    _initialized: bool = False

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    async def initialize(self) -> None:
        if not self._initialized:
            await self.session.initialize()
            self._initialized = True

    async def get_tools(self) -> list[BaseTool]:
        if not self._initialized:
            await self.initialize()

        tools = await self.session.list_tools()
        return [
            MCPTool(
                toolkit=self,
                name=tool.name,
                description=tool.description or "",
                args_schema=create_schema_model(tool.inputSchema),
            )
            for tool in tools.tools
        ]


def create_schema_model(schema: dict[str, t.Any]) -> type[pydantic.BaseModel]:
    """\n    Create a Pydantic model from a JSON schema.\n    """
    class Schema(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

        @classmethod
        def model_json_schema(cls, **kwargs) -> dict[str, t.Any]:
            return schema

    return Schema


class MCPTool(BaseTool):
    """\n    MCP server tool\n    """

    toolkit: MCPToolkit
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True

    async def _arun(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        result = await self.toolkit.session.call_tool(self.name, arguments=kwargs)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

    @property
    def tool_call_schema(self) -> type[pydantic.BaseModel]:
        assert self.args_schema is not None  # noqa: S101
        return self.args_schema