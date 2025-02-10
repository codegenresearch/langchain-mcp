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

    _tools: ListToolsResult | None = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, session: ClientSession):
        super().__init__()
        self.session = session

    async def initialize(self) -> None:
        """
        Initialize the toolkit by setting up the session and fetching tools.

        This method ensures that the session is initialized and the tools are fetched
        from the MCP server. It should be called before using the toolkit.
        """
        if self._tools is None:
            await self.session.initialize()
            self._tools = await self.session.list_tools()

    @t.override
    async def get_tools(self) -> List[BaseTool]:
        """
        Get the list of tools available in the toolkit.

        Returns:
            List[BaseTool]: A list of tools.

        Note:
            The `list_tools` method returns a `PaginatedResult`, but there is no way
            to pass the cursor to retrieve more tools in the current implementation.
        """
        if self._tools is None:
            raise RuntimeError("Must initialize the toolkit first")
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


def create_schema_model(schema: Dict[str, Any]) -> type[pydantic.BaseModel]:
    """
    Create a Pydantic model class from a JSON schema.

    Args:
        schema (Dict[str, Any]): The JSON schema.

    Returns:
        type[pydantic.BaseModel]: A Pydantic model class.
    """
    class Schema(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

        @t.override
        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = pydantic.json_schema.DEFAULT_REF_TEMPLATE,
            schema_generator: type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,
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
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the tool synchronously. This method exists only to satisfy tests.

        Returns:
            Any: The result of the tool execution.

        Warnings:
            This method is deprecated. Use `ainvoke` to run the tool asynchronously.
        """
        warnings.warn(
            "Invoke this tool asynchronously using `ainvoke`. This method exists only to satisfy tests.",
            stacklevel=1,
        )
        return asyncio.run(self._arun(*args, **kwargs))

    @t.override
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the tool asynchronously.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: The result of the tool execution.

        Raises:
            ToolException: If the tool execution results in an error.
        """
        result = await self.session.call_tool(self.name, arguments=kwargs)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

    @t.override
    @property
    def tool_call_schema(self) -> type[pydantic.BaseModel]:
        """
        Get the schema for the tool call.

        Returns:
            type[pydantic.BaseModel]: The schema for the tool call.
        """
        assert self.args_schema is not None  # noqa: S101
        return self.args_schema