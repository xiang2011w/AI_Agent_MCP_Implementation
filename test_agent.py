import os, asyncio
from mcpclient import MCPClient
from agent import Agent
import sys


async def main():
    print("Step 1: Creating MCPClient(s)...")
    fetch_script = os.environ.get(
        "MCP_FETCH_SCRIPT_PATH",
        "/Users/xiang2011w/Desktop/personal/JOB/ALL_IN_AI/MCP_RAG/fetch-mcp/dist/index.js",
    )
    fetch_mcp = MCPClient(
        name="fetch",
        command="node",
        args=[fetch_script],
    )

    current_dir = os.getcwd()
    file_mcp = MCPClient(
        name="file",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", current_dir],
    )
    print("→ Connecting to fetch MCP server…")
    await fetch_mcp.connect_to_server()
    print("DEBUG: fetch_mcp is of type", type(fetch_mcp))
    print("DEBUG: fetch_mcp =", fetch_mcp)

    print("Step 2: Creating Agent...")
    agent = Agent(
        "gpt-4o-mini",
        [fetch_mcp, file_mcp],
        (
            "You are a helpful assistant. "
            "When you need the raw text of a web page, call the function `fetch_txt` "
            'with an {"url": https://www.lux.camera/what-is-hdr/} argument.'
            "After fetching the text, call the function `write_file` "
            'summarize the fetched text and save it to a new file named "webpage.txt" in the ${current_dir} directory.'
        ),
    )

    print("Step 3: Initializing Agent (await agent.init())...")
    await agent.init()
    print("Step 4: Agent initialized.")

    question = "Fetch the text of https://www.lux.camera/what-is-hdr/"
    print(f"Step 5: Invoking agent with question: {question}")
    response = await agent.chat(question)
    print("Step 6: Agent invoke complete.")
    print("Response:", response)

    print("Step 7: Closing agent (if applicable)...")
    await agent.close()
    print("Step 8: Done.")


if __name__ == "__main__":
    print("Starting test_agent_mcp.py")
    try:
        asyncio.run(main())
    except Exception as e:
        print("Exception occurred:", e, file=sys.stderr)
        import traceback

        traceback.print_exc()
