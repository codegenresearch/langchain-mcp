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

    @t.override
    async def get_tools(self) -> list[BaseTool]:  # type: ignore[override]
        await self.initialize()

        return [
            MCPTool(
                toolkit=self,
                name=tool.name,
                description=tool.description or "",
                args_schema=create_schema_model(tool.inputSchema),
            )
            # list_tools returns a PaginatedResult, but I don't see a way to pass the cursor to retrieve more tools\n            for tool in (await self.session.list_tools()).tools\n        ]\n\n\ndef create_schema_model(schema: dict[str, t.Any]) -> type[pydantic.BaseModel]:\n    # Create a new model class that returns our JSON schema.\n    # LangChain requires a BaseModel class.\n    class Schema(pydantic.BaseModel):\n        model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)\n\n        @t.override\n        @classmethod\n        def model_json_schema(\n            cls,\n            by_alias: bool = True,\n            ref_template: str = pydantic.json_schema.DEFAULT_REF_TEMPLATE,\n            schema_generator: type[pydantic.json_schema.GenerateJsonSchema] = pydantic.json_schema.GenerateJsonSchema,\n            mode: pydantic.json_schema.JsonSchemaMode = "validation",\n        ) -> dict[str, t.Any]:\n            return schema\n\n    return Schema\n\n\nclass MCPTool(BaseTool):\n    """\n    MCP server tool\n    """\n\n    toolkit: MCPToolkit\n    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True\n\n    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:\n        warnings.warn(\n            "Invoke this tool asynchronously using `ainvoke`. This method exists only to satisfy tests.", stacklevel=1\n        )\n        return asyncio.run(self._arun(*args, **kwargs))\n\n    async def _arun(self, *args: t.Any, **kwargs: t.Any) -> t.Any:\n        result = await self.toolkit.session.call_tool(self.name, arguments=kwargs)\n        content = pydantic_core.to_json(result.content).decode()\n        if result.isError:\n            raise ToolException(content)\n        return content\n\n    @t.override\n    @property\n    def tool_call_schema(self) -> type[pydantic.BaseModel]:\n        assert self.args_schema is not None  # noqa: S101\n        return self.args_schema