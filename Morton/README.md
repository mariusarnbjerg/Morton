To run the system, the following model: qwen3:8b, must be pulled from Ollama and augmented with the modelfile attached to the project.

To pull from Ollama open CMD and type: 

**ollama pull qwen3:8b**

To build the required model from that type: 

**ollama create morton -f C:\development\gitrep\Morton\Morton\Modelfile** (Path to modelfile)