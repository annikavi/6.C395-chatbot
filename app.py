import gradio as gr
from src.chat import Chatbot


def create_chatbot_interface() -> gr.ChatInterface:
    chatbot = Chatbot()

    def chat(message: str, history: list) -> str:
        return chatbot.get_response(message, history)

    return gr.ChatInterface(
        fn=chat,
        title="MIT Course Catalog Assistant",
        description=(
            "Ask me anything about MIT courses — distribution requirements, "
            "prerequisites, scheduling, or course comparisons. "
            "Powered by real data from [student.mit.edu/catalog](https://student.mit.edu/catalog/index.cgi)."
        ),
        examples=[
            "I'm a 6-3 junior. I still need a CI-H and I'm interested in AI ethics. Any afternoon options?",
            "What are the prerequisites for 6.3900?",
            "I need to fulfill my REST requirement. What options work well with a CS background?",
            "I'm interested in systems security. What courses should I take after 6.1800?",
            "What CI-H courses relate to technology or computing?",
            "I want to take NLP courses. What's the recommended sequence?",
            "Can you compare 6.4100 and 6.3900? Which should I take first?",
            "I'm a TPP student. What CRE should I take?",
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo = create_chatbot_interface()
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))
