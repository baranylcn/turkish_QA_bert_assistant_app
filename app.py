import streamlit as st
from model_loader import load_model_and_tokenizer
import torch


class Assistant:
    def __init__(self):
        # Load content from file
        with open("content.txt", "r", encoding="utf-8") as file:
            self.content = file.read()
        # Load BERT model and tokenizer for Turkish Question Answering
        self.tokenizer, self.model = load_model_and_tokenizer("lserinol/bert-turkish-question-answering")

    def initialize(self):
        if "text_area" not in st.session_state:
            st.session_state.text_area = ""

    def get_answer(self, question):
        # Tokenize the question and content for input to the model
        inputs = self.tokenizer(question, self.content, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        # Get start and end logits from the model
        start_logits = self.model(**inputs)["start_logits"]
        end_logits = self.model(**inputs)["end_logits"]

        # Find the answer span based on logits
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits) + 1

        # Convert answer tokens to string
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer

    def run(self):
        # Set page title for the app
        st.set_page_config(page_title="XXXXX Genel Bilgi Asistanı")
        st.title("XXXXX Genel Bilgi Asistanı")
        self.initialize()
        # Input field for user to input question
        question = st.text_input("Soru", value="", max_chars=None, key=None)
        if st.button("Sor"):
            if question:
                # Get and display answer
                answer = self.get_answer(question)
                st.write(f"Asistan: {answer.capitalize()}")
            else:
                st.write("Lütfen bir soru giriniz.")


if __name__ == "__main__":
    # Instantiate Assistant class and run the application
    assistant = Assistant()
    assistant.run()
