import json
from chatopenai import ChatOpenAI # Make sure chatopenai.py is in the same directory or your PYTHONPATH

# --- Mock Tool Implementation ---
def get_current_weather(location: str, unit: str = "celsius"):
    """Simulates getting current weather. This is our 'tool'."""
    print(f"--- Tool Executing: get_current_weather(location='{location}', unit='{unit}') ---")
    if "boston" in location.lower():
        return json.dumps({"location": location, "temperature": "10", "unit": unit, "forecast": "likely snowy"})
    elif "tokyo" in location.lower():
        return json.dumps({"location": location, "temperature": "25", "unit": unit, "forecast": "sunny with a chance of robots"})
    else:
        return json.dumps({"location": location, "temperature": "22", "unit": unit, "forecast": "pleasant"})

# --- Tool Definition for OpenAI ---
WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["location"],
        },
    },
}

def test_simple_chat(model_name="gpt-3.5-turbo"):
    print("\n--- Test Case: Simple Chat ---")
    chat_bot = ChatOpenAI(model_name=model_name)

    prompt = "Hello! Tell me a short joke about programming."
    print(f"\nUser: {prompt}")
    print("Assistant (streaming):")
    # The .chat() method in your class streams output and returns final content + tool_calls
    # chatopenai.py now returns tool_calls as a list
    content, tool_calls_list = chat_bot.chat(prompt)
    
    print("\n--- Stream Ended ---")
    print(f"Final Accumulated Content: {content}")
    if tool_calls_list: # Check if the list is not empty
        print(f"Tool Calls Made: {tool_calls_list}")
    else:
        print("No Tool Calls Made.")
    
    print("\nMessage History (ChatOpenAI instance):")
    for i, msg in enumerate(chat_bot.messages):
        print(f"  {i}: {msg}")

def test_chat_with_tool_usage(model_name="gpt-3.5-turbo"):
    print("\n--- Test Case: Chat with Tool Usage ---")
    chat_bot = ChatOpenAI(
        model_name=model_name,
        tools=[WEATHER_TOOL_DEFINITION] # Provide the tool definition
    )

    prompt = "What's the weather like in Boston today?"
    print(f"\nUser: {prompt}")
    print("Assistant (streaming first response / tool call):")
    # chatopenai.py now returns tool_calls as a list
    content, tool_calls_list = chat_bot.chat(prompt) 

    print("\n--- Stream Ended (Initial Response) ---")
    print(f"Initial Content: {content if content else '(No direct text content from LLM, expected tool call)'}")
    
    if not tool_calls_list: # Check if the list is empty
        print("No tool calls were made, but expected for this prompt. Test might not proceed as planned.")
        return

    print(f"Tool Calls Received (list): {tool_calls_list}")
    
    # Process tool calls
    # Iterate over the list of tool call dictionaries
    for call_idx, tool_call_item in enumerate(tool_calls_list):
        tool_id = tool_call_item.get("id")
        
        function_details = tool_call_item.get("function")
        if not isinstance(function_details, dict):
            print(f"  Warning: Skipping tool call at index {call_idx} due to missing or invalid 'function' details: {tool_call_item}")
            continue

        tool_name = function_details.get("name")
        tool_arguments_str = function_details.get("arguments")

        if not tool_id or not tool_name:
             print(f"  Warning: Skipping tool call at index {call_idx} due to missing 'id' or 'function.name': {tool_call_item}")
             continue

        print(f"\nProcessing Tool Call (Index {call_idx}):")
        print(f"  ID: {tool_id}")
        print(f"  Name: {tool_name}")
        print(f"  Arguments (raw string): {tool_arguments_str}")

        if tool_name == "get_current_weather":
            try:
                # Parse the string arguments into a Python dictionary
                args = json.loads(tool_arguments_str)
                location = args.get("location")
                unit = args.get("unit", "celsius") # Default if not provided by LLM

                if not location:
                    print("Error: 'location' missing in tool arguments.")
                    tool_result_str = json.dumps({"error": "Location parameter is required."})
                else:
                    # Execute our mock tool
                    tool_result_str = get_current_weather(location=location, unit=unit)
            
            except json.JSONDecodeError:
                print(f"Error: Could not decode tool arguments JSON: {tool_arguments_str}")
                tool_result_str = json.dumps({"error": "Invalid JSON in arguments."})
            except Exception as e:
                print(f"Error during tool execution: {e}")
                tool_result_str = json.dumps({"error": str(e)})
            
            print(f"  Tool Result: {tool_result_str}")
            # Append the tool's result to the message history
            chat_bot.appendToolCallResult(toolCallId=tool_id, result=tool_result_str)
            print(f"Appended tool result for ID {tool_id} to messages.")
        else:
            print(f"  Warning: Unknown tool '{tool_name}'. Skipping.")
            # Optionally, you could append an error message for unknown tools
            # chat_bot.appendToolCallResult(toolCallId=tool_id, result=json.dumps({"error": f"Tool '{tool_name}' not found."}))

    # Pass an empty prompt, or a specific instruction if needed.
    # The LLM will use the tool results from the message history.
    print("\nUser: (Continuing conversation after tool execution)") # Implicitly an empty prompt to .chat()
    print("Assistant (streaming final response):")
    # subsequent_tool_calls will also be a list
    final_content, subsequent_tool_calls_list = chat_bot.chat("") # Empty prompt to summarize
    
    print("\n--- Stream Ended (Final Response) ---")
    print(f"Final Accumulated Content (after tool use): {final_content}")
    if subsequent_tool_calls_list: # Check if the list is not empty
        print(f"Further Tool Calls Made: {subsequent_tool_calls_list}")
    else:
        print("No further tool calls made.")

    print("\nMessage History (ChatOpenAI instance):")
    for i, msg in enumerate(chat_bot.messages):
        print(f"  {i}: {msg}")

if __name__ == "__main__":
    # Ensure your OPENAI_API_KEY is in a .env file for chatopenai.py to load.
    # You can change the model_name if you prefer (e.g., "gpt-4o-mini", "gpt-4-turbo")
    
    test_simple_chat(model_name="gpt-3.5-turbo")
    print("\n" + "="*70 + "\n")
    test_chat_with_tool_usage(model_name="gpt-3.5-turbo") 