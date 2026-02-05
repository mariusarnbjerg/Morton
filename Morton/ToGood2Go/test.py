import ollama

history=[]

# Choose a chat-capable model (ensured it is pulled)
model = 'llama3.1' #jobautomation/OpenEuroLLM-Danish:latest'

while True:
    prompt = input("You: ").strip()
    if prompt.lower() == 'exit' or prompt.lower() == 'quit':
        print("Goodbye")
        break

    message = {
        "role":"user",
        "content": prompt
    }

    history.append(message)
    print("Bot: ", end="", flush=True)

    bot_message_content = ""

    response = ollama.chat(model=model, messages=history, stream=True)

    for chunk in response:
        bot_message_content += chunk.message.content
        print(chunk.message.content, end="", flush=True)

    print()

    bot_message = {
        "role":"assistant",
        "content": bot_message_content
    }
    history.append(bot_message)
