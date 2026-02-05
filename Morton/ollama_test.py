
import ollama

prompt = "why is the sky blue?"

output = ollama.generate(model="llama3.1", prompt=prompt, stream=True)

print(output)
