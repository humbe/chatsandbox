from mlx_lm import load, stream_generate

from tools import call_tool, get_tool_instructions

model, tokenizer = load("mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit")

messages = [
    {
        "role": "system",
        "content": get_tool_instructions()
    },
]
tool_response = ""
while True:

    if len(tool_response) == 0:
        request = input("> ")

        if request == "/q":
            break

        messages.append({
            "role": "user",
            "content": request
        })
    else:
        tool_response = ""

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    thinking=True
    found_tool_call = False
    in_json = False
    tool_call = ""
    full_response = ""

    for response in stream_generate(model, tokenizer, prompt=prompt):
        print(response.text, end="", flush=True)
        full_response += response.text

        if "</think>" in response.text:
            thinking = False

        if not thinking and "{" in response.text:
            in_json = True

        if in_json:
            tool_call += response.text

        if tool_call.startswith("{\"call\":"):
            found_tool_call = True
        elif len(tool_call) > 10:
            # this is not a tool call.
            in_json = False
            tool_call = ""
    print()

    messages.append({
        "role": "assistant",
        "content": full_response
    })

    if found_tool_call:
        tool_response = call_tool(tool_call)

        if len(tool_response) > 0:
            print(tool_response)
            messages.append({
                "role": "tool",
                "content": tool_response
            })
