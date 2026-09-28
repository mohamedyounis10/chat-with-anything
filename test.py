from wraaper import KaggleMistralLLM

llm = KaggleMistralLLM(api_url="https://stinging-gruffly-progress.ngrok-free.dev")
result = llm.invoke("Say only the word: OK")
print(result)