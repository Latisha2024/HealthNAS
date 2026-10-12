from ollama import chat


class HealthcareLLM:
    """Interface for the Qwen model running through Ollama."""

    def __init__(self):
        # Qwen model used for architecture generation
        self.model_name = "qwen2.5:0.5b"

        print("\nInitializing Healthcare LLM...")
        print(f"Using Ollama model: {self.model_name}")

    def generate(self, prompt):
        # Send the project prompt to Qwen
        print("\nGenerating NAS architecture using Qwen...")

        response = chat(
            model=self.model_name,

            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI system for:

LLM-Guided Neural Architecture Search
for Domain-Specific AI.

The selected domain is HEALTHCARE.

Analyze the healthcare machine-learning problem
and generate a neural architecture search space.

The LLM should guide NAS by proposing architecture
parameters that can be explored.

Do not recommend only one fixed architecture.
Provide multiple choices for NAS.

Possible parameters include:

- number of convolutional blocks
- number of filters
- kernel size
- activation function
- pooling
- batch normalization
- dropout
- attention

Do not provide medical diagnosis, treatment,
or medical advice.

Return ONLY valid JSON.
"""
                },
                {
                    # The actual healthcare problem is provided here
                    "role": "user",
                    "content": prompt
                }
            ],

            # Force the model response to JSON format
            format="json",

            # Low temperature gives more consistent output
            options={
                "temperature": 0.1
            }
        )

        # Extract the generated text from the Ollama response
        result = response["message"]["content"]

        return result.strip()