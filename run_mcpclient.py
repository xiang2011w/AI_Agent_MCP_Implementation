import os
import asyncio
from dotenv import load_dotenv
import pprint

load_dotenv()

from mcpclient import MCPClient


async def main():
    # 1) Figure out how to launch your real MCP stdio‐server
    server_script = os.getenv("MCP_SERVER_SCRIPT_PATH")
    if not server_script:
        print("ERROR: MCP_SERVER_SCRIPT_PATH is not set")
        return

    # 2) Determine launcher based on extension and instantiate
    if server_script.endswith(".py"):
        command = "python"
        args = [server_script]
    elif server_script.endswith(".js"):
        command = "node"
        args = [server_script]
    else:
        print("ERROR: Unsupported server script type:", server_script)
        return

    client = MCPClient(
        name="real‐mcp",
        command=command,
        args=args,
    )
    print("✅ Instantiated MCPClient")
    print("  name   :", client.name)
    print("  command:", client.command)
    print("  args   :", client.args)
    print("  version:", client.version)

    # 3) Connect and pull down the real tool list
    await client.connect_to_server()
    tools = client.get_tools()
    print("✅ Retrieved tools from server:")
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    # 3.2) Full metadata dump
    print("\n🔍 Full tool metadata:")
    full = []
    for t in tools:
        if hasattr(t, "to_dict"):
            full.append(t.to_dict())
        elif hasattr(t, "__dict__"):
            full.append(t.__dict__)
        else:
            full.append({"raw": str(t)})
    pprint.pprint(full, indent=2)

    # 4) Tear down
    await client.disconnect_from_server()
    print("✅ disconnected cleanly")


if __name__ == "__main__":
    asyncio.run(main())
