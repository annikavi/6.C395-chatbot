from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN

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
        
    def format_prompt(self, user_input):
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
        return user_input
        
    def get_response(self, user_input):
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
        # TODO: use tools to go to the course catalog
        try:
            formatted_input = self.format_prompt(user_input)
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in MIT Course Catalog. \
                    The catalog contains hundreds of courses, each with prerequisites, schedules, \
                    departments, distribution requirements (CI-H, HASS, REST), instructors, \
                    and class formats. A student might say “I’m a 6-3 junior who needs a CI-H, \
                    prefers afternoon classes, and am interested in AI ethics” and you need to reason \
                    across all of those dimensions. You should help students find courses that match \
                    their constraints and interests. Do not try to immediately give suggestions. If the instructions are unclear, \
                    ask more questions to be able to give better suggestions"},
                {"role": "user", "content": formatted_input},
            ]
            output = self.client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=messages,
                        max_tokens=1024,
                    )
            response = output.choices[0].message.content

            return response
        except Exception as e:
            return f"Error: {str(e)}"
