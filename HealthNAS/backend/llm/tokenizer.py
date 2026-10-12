from transformers import AutoTokenizer


class HealthcareTokenizer:
    """Tokenizer for demonstrating the Qwen tokenization process."""

    def __init__(self):
        # Qwen tokenizer used by the project
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        print("\nLoading Qwen tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        print("Qwen tokenizer loaded.")

    def tokenize(self, text):
        """Convert text into tokens."""

        return self.tokenizer.tokenize(text)

    def token_ids(self, text):
        """Convert text into numerical token IDs."""

        encoded = self.tokenizer(
            text,
            add_special_tokens=True
        )

        return encoded["input_ids"]

    def demonstrate(self, text):
        """Display the tokenization process."""

        # Convert the input text into tokens
        tokens = self.tokenize(text)

        # Convert the tokens into numerical IDs
        ids = self.token_ids(text)

        print("\n")
        print("=" * 70)
        print("LLM TOKENIZATION DEMONSTRATION")
        print("=" * 70)

        print(f"\nInput:\n{text}")

        print("\nTokens:")
        print(tokens)

        print("\nToken IDs:")
        print(ids)

        print(
            f"\nNumber of tokens: {len(tokens)}"
        )

        return {
            "tokens": tokens,
            "token_ids": ids
        }