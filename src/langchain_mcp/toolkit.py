# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import warnings
from collections.abc import Callable
from typing import Any, Dict, List, Union

import pydantic
import pydantic_core
import typing_extensions as t
from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException
from mcp import ClientSession, ListToolsResult, Tool


class MCPToolkit(BaseToolkit):
    """
    MCP server toolkit
    """

    session: ClientSession
    """The MCP session used to obtain the tools"""

    _tools: List[BaseTool] = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, session: ClientSession):
        super().__init__()
        self.session = session

    async def initialize(self) -> None:
        if self._tools is None:
            await self.session.initialize()
            tools = (await self.session.list_tools()).tools
            self._tools = [
                MCPTool(
                    toolkit=self,
                    session=self.session,
                    name=tool.name,
                    description=tool.description or "",
                    args_schema=create_schema_model(tool.inputSchema),
                )
                for tool in tools
            ]

    @t.override
    async def get_tools(self) -> list[BaseTool]:
        if self._tools is None:
            raise RuntimeError("Toolkit has not been initialized. Call `initialize` method first.")
        return self._tools


def create_schema_model(schema: Dict[str, Any]) -> t.type[pydantic.BaseModel]:
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
            schema_generator: t.type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,
            mode: pydantic.json_schema.JsonSchemaMode = "validation",
        ) -> Dict[str, Any]:
            return schema

    return Schema


class MCPTool(BaseTool):
    """
    MCP server tool
    """

    toolkit: MCPToolkit
    session: ClientSession
    handle_tool_error: Union[bool, str, Callable[[ToolException], str], None] = True

    @t.override
    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        warnings.warn(
            "Invoke this tool asynchronously using `ainvoke`. This method exists only to satisfy tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    @t.override
    async def _arun(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        result = await self.session.call_tool(self.name, arguments=kwargs)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

    @t.override
    @property
    def tool_call_schema(self) -> t.type[pydantic.BaseModel]:
        assert self.args_schema is not None  # noqa: S101
        return self.args_schema