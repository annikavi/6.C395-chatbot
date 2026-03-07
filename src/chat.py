from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN

class Chatbot:
    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        
    def format_prompt(self, user_input):
        """
        Formats the input using Llama 3.1 chat templates to ensure the 
        model stays in 'Advisor Mode'.
        """
        system_instructions = (
            "You are the MIT Course Navigator, an expert academic advisor. "
            "Your goal is to help students find courses by reasoning across multiple dimensions:"
            "Department/Major (e.g., Course 6), Prerequisites, Distribution Requirements (CI-H, HASS, REST), "
            "Class Formats, and Scheduling. "
            "\n\nStrict Rules:\n"
            "1. If a student mentions a specific requirement (like CI-H), prioritize those.\n"
            "2. Always check for prerequisite conflicts if the student mentions their year.\n"
            "3. If timing is mentioned (e.g., 'afternoon'), filter results accordingly.\n"
            "4. Be concise, professional, and encouraging."
        )
        
        # Llama 3.1 Chat Format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instructions}<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
        
    def get_response(self, user_input):
        """
        Generates a response using the formatted prompt.
        """
        formatted_input = self.format_prompt(user_input)
        
        # We use simple text_generation here; 
        # max_new_tokens controls how long the advice can be.
        response = self.client.text_generation(
            formatted_input,
            max_new_tokens=500,
            temperature=0.7,  # Balanced between creative and factual
            stop=["<|eot_id|>"]
        )
        
        return response
