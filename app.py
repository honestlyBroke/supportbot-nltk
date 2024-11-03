import streamlit as st
from chatbot import load_data, train_model, predict_intent, respond_to_intent

# Load and train the model once at the start of the app
data = load_data()
model = train_model(data)

if model is None:
    st.error("Model could not be trained. Please check the logs for more information.")
else:
    # Streamlit UI
    st.set_page_config(page_title="Support Bot", page_icon=":robot_face:")
    st.title("Customer Support Chatbot")
    st.write("Welcome to our Customer Support Chatbot! Please select a category and then a question.")

    # Categorized list of questions
    categories = {
        "Account Management": [
            "How can I reset my password?",
            "I forgot my password.",
            "How do I delete my account?",
            "Can I change my subscription plan?",
            "Is there a way to recover my account?",
            "What happens if I miss a payment?",
            "Can I pause my subscription?"
        ],
        "Order Assistance": [
            "Can I track my order?",
            "How do I report a problem with my order?",
            "What do I do if I receive a damaged item?",
            "How can I change the shipping address on my order?",
            "What should I do if my order is delayed?",
            "How do I find my order number?"
        ],
        "Payment & Billing": [
            "How do I update my billing information?",
            "What payment methods do you accept?",
            "Can you help me with a refund?",
            "Are there any discounts available?",
            "Do you offer gift cards?",
        ],
        "Support & Feedback": [
            "What are your support hours?",
            "How can I contact customer support?",
            "Do you offer live chat support?",
            "How can I provide feedback about your service?",
            "What is your privacy policy?",
            "Can I subscribe to your newsletter?",
            "How can I unsubscribe from emails?"
        ],
        "Policies": [
            "What is your return policy?",
            "What is your warranty policy?",
            "Can I speak to a manager?",
        ],
        "Product Information": [
            "Where can I find product documentation?",
        ]
    }

    # Dropdown for user to select a category
    selected_category = st.selectbox("Select a category:", list(categories.keys()))

    # Dropdown for user to select a question based on the selected category
    selected_question = st.selectbox("Select a question:", categories[selected_category])

    if st.button("Ask"):
        if selected_question:
            # Get the predicted intent and response
            intent = predict_intent(model, selected_question)
            response = respond_to_intent(intent)

            # Display bot's response as a success message
            st.success(f"**Bot:** {response}")
        else:
            st.warning("Please select a question.")

