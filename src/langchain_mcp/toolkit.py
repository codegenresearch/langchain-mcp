# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import warnings
from collections.abc import Callable
from typing import Any, Optional, Type

import pydantic
import pydantic_core
import typing_extensions as t
from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException
from mcp import ClientSession, ListToolsResult, Tool
from mcp.types import CallToolResult, TextContent


class MCPToolkit(BaseToolkit):
    """
    MCP server toolkit
    """

    session: ClientSession
    """The MCP session used to obtain the tools"""

    _tools: ListToolsResult | None = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, session: ClientSession):
        super().__init__()
        self.session = session

    @t.override
    async def initialize(self) -> None:
        """
        Initialize the toolkit by fetching tools from the MCP session.
        """
        if self._tools is None:
            await self.session.initialize()
            self._tools = await self.session.list_tools()

    @t.override
    async def get_tools(self) -> list[BaseTool]:
        if self._tools is None:
            raise RuntimeError("Must initialize the toolkit first.")
        return [
            MCPTool(
                toolkit=self,
                session=self.session,
                name=tool.name,
                description=tool.description or "",
                args_schema=create_schema_model(tool.inputSchema),
            )
            for tool in self._tools.tools
        ]


def create_schema_model(schema: dict[str, t.Any]) -> type[pydantic.BaseModel]:
    # Create a new model class that returns our JSON schema.
    # LangChain requires a BaseModel class.
    class Schema(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

        @t.override
        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = pydantic.json_schema.DEFAULT_REF_TEMPLATE,
            schema_generator: Type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,
            mode: pydantic.json_schema.JsonSchemaMode = "validation",
        ) -> dict[str, t.Any]:
            return schema

    return Schema


class MCPTool(BaseTool):
    """
    MCP server tool
    """

    toolkit: MCPToolkit
    session: ClientSession
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True

    def __init__(
        self,
        toolkit: MCPToolkit,
        session: ClientSession,
        name: str,
        description: str,
        args_schema: type[pydantic.BaseModel],
    ):
        super().__init__(name=name, description=description, args_schema=args_schema)
        self.toolkit = toolkit
        self.session = session

    @t.override
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "Invoke this tool asynchronously using `ainvoke`. This method exists only to satisfy standard tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    @t.override
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        result: CallToolResult = await self.session.call_tool(self.name, arguments=kwargs)
        content: str = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

    @t.override
    @property
    def tool_call_schema(self) -> type[pydantic.BaseModel]:
        assert self.args_schema is not None  # noqa: S101
        return self.args_schema