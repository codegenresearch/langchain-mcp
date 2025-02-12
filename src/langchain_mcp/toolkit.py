# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import warnings
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Type, Union

import pydantic
from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException
from mcp import ClientSession, ListToolsResult, Tool
from mcp.types import CallToolResult, TextContent
from mcp.types import ListToolsResult, Tool

class MCPToolkit(BaseToolkit):
    """\n    MCP server toolkit\n    """

    session: ClientSession
    """The MCP session used to obtain the tools"""

    _initialized: bool = False

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, session: ClientSession):
        self.session = session

    async def initialize(self) -> None:
        if not self._initialized:
            await self.session.initialize()
            self._initialized = True

    async def get_tools(self) -> List[BaseTool]:
        await self.initialize()
        tools_result: ListToolsResult = await self.session.list_tools()
        return [
            MCPTool(
                toolkit=self,
                name=tool.name,
                description=tool.description or "",
                args_schema=self._create_schema_model(tool.inputSchema),
            )
            for tool in tools_result.tools
        ]

    def _create_schema_model(self, schema: Dict[str, Any]) -> Type[pydantic.BaseModel]:
        class Schema(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

            @classmethod
            def model_json_schema(
                cls,
                by_alias: bool = True,
                ref_template: str = pydantic.json_schema.DEFAULT_REF_TEMPLATE,
                schema_generator: Type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,
                mode: str = "validation",
            ) -> Dict[str, Any]:
                return schema

        return Schema


class MCPTool(BaseTool):
    """\n    MCP server tool\n    """

    toolkit: MCPToolkit
    handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = True

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "Invoke this tool asynchronously using `ainvoke`. This method exists only to satisfy tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        result: CallToolResult = await self.toolkit.session.call_tool(self.name, arguments=kwargs)
        content: str = pydantic.json.loads(pydantic.json.dumps(result.content))
        if result.isError:
            raise ToolException(content)
        return content

    @property
    def tool_call_schema(self) -> Type[pydantic.BaseModel]:
        assert self.args_schema is not None  # noqa: S101
        return self.args_schema