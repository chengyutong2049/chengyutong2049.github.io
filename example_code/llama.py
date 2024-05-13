from transformers import pipeline

# Create a pipeline for text generation using the Llama-2-70b-hf model
pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-hf")

# Use the pipeline to generate text
# Replace "Your prompt here" with your actual prompt
prompt = "Your prompt here"
generated_text = pipe(prompt, max_length=50)  # You can adjust max_length as needed

# Print the generated text
print(generated_text[0]['generated_text'])