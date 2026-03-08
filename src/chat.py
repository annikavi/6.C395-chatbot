from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from typing import List, Dict
class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.MAX_TOKENS = 500

    
    def format_messages(self, prompt: str, history: List[Dict]) -> List[Dict]:
     
        messages = [{"role": "system", "content": "You are a helpful assistant that specializes in helping students navigate the MIT course catalog."}]
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": prompt})
        return messages

    def format_prompt(self, user_input: str)->str:
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question

        Returns:
            str: A formatted prompt ready for the model
        
        Example prompt format:
            "You are a helpful assistant that specializes in...
             User: {user_input}
             Assistant:"

        """
 
        placeholder_prompt = f"""
        User: {user_input}
        """
        return placeholder_prompt
        
    def get_response(self, user_input, history=[])->str:
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """

        prompt = self.format_prompt(user_input)

        messages = self.format_messages(prompt, history)
        response = self.client.chat_completion(messages=messages, max_tokens=self.MAX_TOKENS)

        if not response.choices[0].message.content:
            return "Sorry! Me cant help you :("
        return response.choices[0].message.content

