# Copyright (C) 2024 Andrew Wason # SPDX-License-Identifier: MIT import asyncio import warnings from collections.abc import Callable import pydantic import pydantic_core import typing_extensions as t from langchain_core.tools.base import BaseTool, BaseToolkit, ToolException from mcp import ClientSession, ListToolsResult, Tool from mcp.types import CallToolResult, TextContent class MCPToolkit(BaseToolkit):     """     MCP server toolkit     This class provides a toolkit for interacting with the MCP server. It initializes a session with the MCP server and retrieves tools available for use.     """     session: ClientSession     _initialized: bool = False     _tools: list[BaseTool] = []     def __init__(self, session: ClientSession):         self.session = session     async def initialize(self) -> None:         """         Initialize the MCPToolkit by setting up the session and retrieving tools.         """         if not self._initialized:             await self.session.initialize()             self._tools = await self.get_tools()             self._initialized = True     async def get_tools(self) -> list[BaseTool]:         """         Retrieve tools from the MCP server.         Returns:             list[BaseTool]: A list of tools available from the MCP server.         Raises:             RuntimeError: If the toolkit has not been initialized.         """         if not self._initialized:             raise RuntimeError("MCPToolkit has not been initialized. Call initialize() first.")         tools_result: ListToolsResult = await self.session.list_tools()         self._tools = [             MCPTool(                 toolkit=self,                 name=tool.name,                 description=tool.description or "",                 args_schema=self.create_schema_model(tool.inputSchema),             )             for tool in tools_result.tools         ]         return self._tools     @staticmethod     def create_schema_model(schema: dict[str, t.Any]) -> type[pydantic.BaseModel]:         """         Create a Pydantic model class from a JSON schema.         Args:             schema (dict[str, t.Any]): The JSON schema to convert into a Pydantic model.         Returns:             type[pydantic.BaseModel]: A Pydantic model class representing the JSON schema.         """         class Schema(pydantic.BaseModel):             model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)             @classmethod             def model_json_schema(                 cls,                 by_alias: bool = True,                 ref_template: str = pydantic.json_schema.DEFAULT_REF_TEMPLATE,                 schema_generator: type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,                 mode: pydantic.json_schema.JsonSchemaMode = "validation",             ) -> dict[str, t.Any]:                 return schema         return Schema class MCPTool(BaseTool):     """     MCP server tool     This class represents a tool available from the MCP server. It handles the invocation of the tool and manages its execution.     """     toolkit: MCPToolkit     name: str     description: str     args_schema: type[pydantic.BaseModel]     handle_tool_error: bool | str | Callable[[ToolException], str] | None = True     def __init__(self, toolkit: MCPToolkit, name: str, description: str, args_schema: type[pydantic.BaseModel]):         self.toolkit = toolkit         self.name = name         self.description = description         self.args_schema = args_schema     def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:         """         Invoke the tool synchronously.         This method is provided to satisfy tests but should be used asynchronously with `ainvoke`.         Args:             *args (t.Any): Positional arguments for the tool.             **kwargs (t.Any): Keyword arguments for the tool.         Returns:             t.Any: The result of the tool execution.         """         warnings.warn(             "Invoke this tool asynchronously using `ainvoke`. This method exists only to satisfy tests.", stacklevel=1         )         return asyncio.run(self._arun(*args, **kwargs))     async def _arun(self, *args: t.Any, **kwargs: t.Any) -> t.Any:         """         Invoke the tool asynchronously.         Args:             *args (t.Any): Positional arguments for the tool.             **kwargs (t.Any): Keyword arguments for the tool.         Returns:             t.Any: The result of the tool execution.         Raises:             ToolException: If the tool execution results in an error.         """         try:             result: CallToolResult = await self.toolkit.session.call_tool(self.name, arguments=kwargs)             content: str = pydantic_core.to_json(result.content).decode()             if result.isError:                 raise ToolException(content)             return content         except Exception as e:             raise ToolException(f"An error occurred while executing the tool: {e}")     @property     def tool_call_schema(self) -> type[pydantic.BaseModel]:         """         Get the schema for the tool call.         Returns:             type[pydantic.BaseModel]: The schema for the tool call.         """         assert self.args_schema is not None  # noqa: S101         return self.args_schema