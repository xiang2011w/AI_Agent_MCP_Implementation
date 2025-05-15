# MCP Clients and MCP server maintain 1:1 connections
# inside the host application like LLM applications (Claude Desktop or IDEs) which initiate connections
# https://modelcontextprotocol.io/docs/concepts/architecture

# reference: https://modelcontextprotocol.io/quickstart/client
import asyncio
from typing import Optional, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


# MCP client launches a separate server process and uses its stdin/stdout as a JSON-RPC channel:
# client handshakes with server via initialize, discovers available commands/tools via list_tools,
# then makes tool calls by sending JSON requests and reading responses.
# When you're done, it cleanly closes the pipes and stops the server.
class MCPClient:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] = [],
        version: str = "0.0.1",
        tools: list[str] = [],
    ) -> None:
        self.session: Optional[ClientSession] = None
        # AsyncExitStack is a context manager that manages a stack of async context managers.
        # Allows you to "enter" several async resources (e.g. transports, network sessions)
        # and then cleanly "exit" all of them in reverse order when you're done
        self.exit_stack = AsyncExitStack()
        self.name = name
        # command to start a new MCP server process e.g. python -m my_mcp_server --stdio
        # Communicate with this new server process over stdio
        # client = MCPClient(
        #     name="my-agent",
        #     command="python",
        #     args=["-m", "my_mcp_server", "--stdio"],
        # )
        self.command = command
        self.args = args
        self.version = version
        self.tools = tools

    async def connect_to_server(self):
        """
        Connect to an MCP server
        """
        # stdio: standard input/output
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # Handshake with the mcp server
        # await self.session.initialize(name=self.name, version=self.version)
        """Connect to the MCP server"""
        await self.session.initialize()

        # List available tools on the mcp server
        response = await self.session.list_tools()
        # save the server‐defined Tool objects (with name, description, params,…)
        self.tools = response.tools
        print("\nConnected to server with tools:")
        for tool in self.tools:
            print(f"\nTool: {tool.name}")
            print(f"Description: {getattr(tool, 'description', 'N/A')}")
            # parameters may live under .function.parameters in the Pydantic Tool model
            if hasattr(tool, "function") and hasattr(tool.function, "parameters"):
                params = tool.function.parameters
            else:
                params = getattr(tool, "parameters", None)
            print(f"Parameters: {params}")

    def get_tools(self):
        return self.tools

    async def call_tool(self, tool_name: str, tool_params: dict):
        """Call a tool with the given name and arguments.

        Args:
            tool_name (str): The name of the tool to call
            tool_params (dict): The arguments to pass to the tool
        """
        response = await self.session.call_tool(tool_name, tool_params)
        return response

    # disconnect from the server
    async def disconnect_from_server(self):
        """
        Gracefully close the MCP session and exit stack.
        Safe to call even if never connected.
        """
        if self.session is not None:
            # No .close() on ClientSession; exit_stack will clean up both transport and session.
            self.session = None

        # Closing the exit stack
        # The MCP session finishes any in-flight requests and shuts down.
        # The subprocess is terminated.
        # The I/O pipes are closed.
        await self.exit_stack.aclose()
