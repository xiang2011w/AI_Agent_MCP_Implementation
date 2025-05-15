from chatopenai import ChatOpenAI
import json
import asyncio


class Agent:
    def __init__(self, model, mcpClients, sysprompt="", context="") -> None:
        self.mcpClients = mcpClients
        self.model = model
        self.sys_prompt = sysprompt
        self.context = context
        self.llm = None

    async def init(self):
        print("Initializing mcp clients.....")
        for mcp in self.mcpClients:
            if isinstance(mcp, str):
                raise TypeError(f"Expected MCPClient, got str: {mcp}")
            await mcp.connect_to_server()

        # Convert all MCP tools to OpenAI function-calling tools
        all_tools = []
        for client in self.mcpClients:
            for tool in client.get_tools():
                all_tools.append(self.convert_mcp_tool_to_openai_function(tool))

        print("Got all tools ....", all_tools)
        print("Initializing LLM ....")

        self.llm = ChatOpenAI(
            self.model,
            system_prompt=self.sys_prompt,
            tools=all_tools,
            context=self.context,
        )
        print("LLM initialized ....")

    async def close(self):
        print("Closing MCP clients ....")
        for mcp in self.mcpClients:
            try:
                await mcp.disconnect_from_server()
            except asyncio.CancelledError:
                print("Warning: MCP client cleanup cancelled.")
            except Exception as e:
                print(f"Warning: Error closing MCP client: {e}")

    # chat with the LLM agent to make tool calls and get the result
    async def chat(self, prompt: str):
        if not self.llm:
            raise Exception("Agent not initialized")

        content, tool_calls = self.llm.chat(prompt=prompt)
        while True:
            if len(tool_calls) > 0:
                # process all tool calls
                for tool_call in tool_calls:
                    # find the mcp client that handles current tool call
                    mcp = next(
                        (
                            client
                            for client in self.mcpClients
                            if any(
                                t.name == tool_call["function"]["name"]
                                for t in client.get_tools()
                            )
                        ),
                        None,
                    )

                    if mcp:
                        print(f"Calling tool: {tool_call['function']['name']}")
                        print(f"Arguments: {tool_call['function']['arguments']}")
                        # call the tool and get the result
                        result = await mcp.call_tool(
                            tool_call["function"]["name"],
                            json.loads(tool_call["function"]["arguments"]),
                        )

                        # convert the result to a string
                        result_str = ""
                        if hasattr(result, "content") and result.content:
                            # only get the text of the first content
                            # LLM expects the tool result in a JSON-serialized string matching the tool's expected output schema.
                            # If the format does not match what the LLM expects, it may not recognize the tool as complete and will
                            # keep re-calling the tool --> infinite loop
                            result_dict = {
                                "content": (
                                    result.content[0].text if result.content else ""
                                ),
                                "isError": getattr(result, "isError", False),
                            }
                            # convert the result to a string
                            result_str = json.dumps(result_dict)
                        else:
                            result_str = str(result)

                        print(f"Result: {result_str}")
                        self.llm.append_tool_result(tool_call["id"], result_str)
                    else:
                        self.llm.append_tool_result(tool_call["id"], "Tool not found")

                # continue the conversation with the updated context with the LLM
                content, tool_calls = self.llm.chat("")
                continue

            # no tool calls, end the conversation
            await self.close()
            return content

    # convert an MCP tool object to OpenAI function-calling tool schema
    @staticmethod
    def convert_mcp_tool_to_openai_function(tool):
        """Convert an MCP tool object to OpenAI function-calling tool schema."""
        if hasattr(tool, "function") and hasattr(tool.function, "parameters"):
            params = tool.function.parameters
        else:
            params = getattr(tool, "parameters", {})

        if "fileName" in params:
            params["path"] = params.pop("fileName")

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "parameters": params or {},
            },
        }
