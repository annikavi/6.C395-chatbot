"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from src.chat import Chatbot

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    # Initialize your Chatbot instance
    chatbot = Chatbot()
    
    def chat(message, history):
        """
        Generates a response for the MIT Course Catalog assistant.
        """
        try:
            # Call the get_response method from your Chatbot class
            response = chatbot.get_response(message)
            return response
            
        except Exception as e:
            # repr(e) provides the full technical class name and message
            error_detail = repr(e) 
            print(f"FULL DEBUG ERROR: {error_detail}") # Check your terminal for this!
            return f"TECHNICAL ERROR: {error_detail}"

    # Create Gradio interface. 
    # Customized for the MIT Course Catalog context.
    demo = gr.ChatInterface(
        chat,
        title="MIT Course Navigator (6.C395)",
        description=(
            "I'm your AI Academic Advisor. Ask me about MIT courses, "
            "CI-H requirements, or finding classes that fit your 6-3 schedule."
        ),
        examples=[
            "I'm a 6-3 junior who needs a CI-H, prefers afternoon classes, and is interested in AI ethics.",
            "What are some REST requirements for a Course 8 major?",
            "I need a HASS-S course that doesn't have 8:00 AM lectures."
        ],
        # theme="soft" # Optional: gives it a cleaner, modern look
    )
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
