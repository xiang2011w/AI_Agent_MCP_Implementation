from dotenv import (
    load_dotenv,
)  # Import function to load environment variables from a .env file
import os  # Module to access environment variables and OS functions

# import openai                      # OpenAI client library for interacting with OpenAI APIs
from openai import OpenAI  # Import the OpenAI client

load_dotenv()  # Load environment variables from .env into the OS environment


class ChatOpenAI:
    """
    ChatOpenAI provides a chat interface to the OpenAI API.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        system_prompt: str = "",  # e.g. "You are a helpful translator. Translate every user message into French."
        tools: list[
            dict
        ] = None,  # list of tools which are defined as JSON‐style dictionaries (use None for default)
        context: str = "",
    ):
        # Load API key from environment and configure OpenAI client
        self.api_key = os.getenv(
            "OPENAI_API_KEY"
        )  # Retrieve API key from OS environment
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment."
            )

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Store model configuration
        self.model_name = model_name  # The name of the model to use (e.g., "gpt-4")
        self.temperature = temperature  # Sampling temperature for response randomness
        self.system_prompt = system_prompt  # system_prompt, set only once at begining, tell LLM what to do
        self.tools = tools  # what the model can call, Function definitions for function-calling (if any)
        self.context = context  # Initial user context or instructions
        # Initialize the message history
        self.messages: list[dict] = (
            []
        )  # A chronological list of all messages between system, user, assistant
        # self.messages[0] is the very first message you sent (usually the system prompt),
        # and self.messages[-1] is the most recent message in the conversation.

    def chat(self, prompt: str):
        """
        Sends a user prompt to the chat model and streams the response.
        Yields each chunk of the assistant's reply as it arrives.
        Returns the full accumulated content and tool calls at the end.
        """

        # Add system prompt at the beginning of the conversation if provided
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        # Include any pre-existing context as a user message
        if self.context:
            self.messages.append({"role": "user", "content": self.context})

        # Append the current user prompt to the message history
        self.messages.append({"role": "user", "content": prompt})

        # Create a streaming chat completion request
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            tools=self.tools,
            stream=True,
        )

        accumulated_content = ""
        # tool_calls are typically a list directly from the delta
        # list of dictionaries, each dictionary contains the tool call id, type, and function
        current_tool_calls_list = []  # Stores fully formed tool calls from the stream

        # Temporary storage for incrementally built tool calls
        # Keyed by index, stores {'id': ..., 'type': 'function', 'function': {'name': ..., 'arguments': ...}}
        building_tool_calls = {}

        # Loop over each streamed chunk as it arrives
        # When streaming, the API delivers chunks sequentially
        # and in each chunk choices[0] always contains the newest portion of the generated response.
        for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # 1. If the model signals it's done, exit the loop
            if finish_reason:
                print(f"\nStream finished. Reason: {finish_reason}")  # For debugging
                break

            # 2. Handle plain-text increments
            if delta.content:
                text = delta.content
                print(text, end="", flush=True)
                accumulated_content += text

            # 3. Handle function/tool calls sent incrementally
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    index = tool_call_chunk.index

                    if (
                        index not in building_tool_calls
                    ):  # First time we see this tool call index
                        building_tool_calls[index] = {
                            "id": tool_call_chunk.id,  # ID is usually in the first chunk for a tool_call
                            "type": "function",  # Assuming type is function
                            "function": {"name": "", "arguments": ""},
                        }

                    # Update ID if it's newly provided
                    if tool_call_chunk.id and not building_tool_calls[index]["id"]:
                        building_tool_calls[index]["id"] = tool_call_chunk.id

                    if tool_call_chunk.function:
                        if tool_call_chunk.function.name:
                            building_tool_calls[index]["function"][
                                "name"
                            ] = tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments:
                            building_tool_calls[index]["function"][
                                "arguments"
                            ] += tool_call_chunk.function.arguments

        # After stream, finalize any built tool calls
        if building_tool_calls:
            # Convert the dictionary of built tool calls into a list
            # and print them as they are completed.
            for index in sorted(building_tool_calls.keys()):
                finalized_call = building_tool_calls[index]
                current_tool_calls_list.append(finalized_call)
                print(f"\nTool Call (Index {index}): {finalized_call}")

        # Append the model response to assistant's message
        assistant_message = {
            "role": "assistant",
            "content": accumulated_content or None,
        }
        # if any, append tool calls to the assistant's message
        if current_tool_calls_list:
            assistant_message["tool_calls"] = current_tool_calls_list

        # append the assistant's message to the messages
        self.messages.append(assistant_message)

        return accumulated_content, current_tool_calls_list

    def append_tool_result(self, tool_call_id: str, result: str):
        """Append a tool call result to the conversation history"""
        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
        )

    def appendToolCallResult(self, toolCallId: str, result: str):
        """
        Appends a tool call result to the messages.
        """
        self.messages.append(
            {"role": "tool", "content": result, "tool_call_id": toolCallId}
        )
