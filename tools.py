import json
import inspect
import subprocess
from memory_tool import get_memory

class AgentTools:
    def remember(self, memory:str):
        get_memory().store(memory)

    def recall(self, topic:str) -> str:
        return get_memory().search(topic)

    def shell(self, command:str) -> str:
        """Runs a command in the unix shell."""
        print(f"Running {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True)

            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Exit code: {e.returncode} {e.stderr}"

class GuardRail:
    def remember(self, memory:str) -> bool:
        return True

    def recall(self, topic:str) -> bool:
        return True

    def shell(self, command:str) -> bool:
        allow = input(f"Run the command {command}? [y/n]")

        return allow == "y" and not "rm " in command

def call_tool(json_str:str) -> str:
    result = ""
    tools = AgentTools()
    guards = GuardRail()
    try:
        call = json.loads(json_str)

        tool_name = call["call"]
        args = call.get("args", {})

        guard = getattr(guards, tool_name)

        if guard and not guard(**args):
            return "Tool Access Denied"

        tool = getattr(tools, tool_name)
        if not tool:
            return f"Failed to find tool {tool_name}"

        tool_result = tool(**call["args"])
        if tool_result is not None:
            result = str(tool_result)
    except KeyError as e:
        print(f"Invalid call json missing key {e.args[0]}")

    return result

def get_tool_instructions() -> str:
    desc = "The following tools are available to call for more information:\n";
    tools = AgentTools()
    for name in dir(tools):
        if name.startswith("_"):
            continue
        tool = getattr(tools, name)
        if not callable(tool):
            continue

        desc += f"tool_name:{name} params:["

        sig = inspect.signature(tool)
        for param_name, param_obj in sig.parameters.items():
            desc += f"{param_name} "
        if tool.__doc__ != None:
            desc += f"]\n\t({tool.__doc__})\n"
        else:
            desc += "]\n"

    desc += "When calling a tool, output json in this format " \
    "{\"call\": \"tool_name\", \"args\": {\"param_name\": value, ...}}"

    return desc
