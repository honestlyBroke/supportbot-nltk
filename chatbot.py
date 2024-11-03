import nltk
import pandas as pd
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download required NLTK data
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Configure logging
logger.add("chatbot.log", level="INFO", format="{time} {level} {message}", rotation="1 MB")

def load_data(source_type='dict', source=None):
    """
    Load data for training the chatbot.
    
    Args:
        source_type (str): Type of data source ('dict', 'csv', 'json').
        source (str): Path to the source file if applicable.

    Returns:
        pd.DataFrame: DataFrame containing queries and intents.
    """
    try:
        if source_type == 'dict':
            data = pd.DataFrame({
                'query': [
                    "How can I reset my password?",
                    "What are your support hours?",
                    "How to cancel my account?",
                    "Can I track my order?",
                    "What is your return policy?",
                    "I forgot my password.",
                    "Can I change my subscription plan?",
                    "How do I update my billing information?",
                    "Is there a way to recover my account?",
                    "What payment methods do you accept?",
                    "Can you help me with a refund?",
                    "How do I report a problem with my order?",
                    "What do I do if I receive a damaged item?",
                    "How can I contact customer support?",
                    "Do you offer live chat support?",
                    "Where can I find product documentation?",
                    "Can I change the shipping address on my order?",
                    "What should I do if my order is delayed?",
                    "How can I provide feedback about your service?",
                    "Can I pause my subscription?",
                    "What happens if I miss a payment?",
                    "Are there any discounts available?",
                    "How do I delete my account?",
                    "Can I speak to a manager?",
                    "How do I find my order number?",
                    "What is your privacy policy?",
                    "Can I subscribe to your newsletter?",
                    "Do you offer gift cards?",
                    "How can I unsubscribe from emails?",
                    "What is your warranty policy?",
                    "How do I change my email address?"
                ],
                'intent': [
                    "password_reset",
                    "support_hours",
                    "account_cancel",
                    "order_tracking",
                    "return_policy",
                    "password_reset",
                    "subscription_change",
                    "billing_update",
                    "account_recovery",
                    "payment_methods",
                    "refund_help",
                    "report_problem",
                    "damaged_item",
                    "contact_support",
                    "live_chat",
                    "product_documentation",
                    "shipping_address",
                    "order_delay",
                    "feedback",
                    "pause_subscription",
                    "missed_payment",
                    "discounts",
                    "delete_account",
                    "speak_to_manager",
                    "find_order_number",
                    "privacy_policy",
                    "newsletter_subscription",
                    "gift_cards",
                    "unsubscribe_emails",
                    "warranty_policy",
                    "change_email"
                ]
            })
        elif source_type == 'csv':
            data = pd.read_csv(source)
        elif source_type == 'json':
            data = pd.read_json(source)
        else:
            raise ValueError("Unsupported data source type provided.")
        
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['query', 'intent'])

def preprocess_text(text):
    """
    Preprocess the input text for model training.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The processed text.
    """
    try:
        tokens = word_tokenize(text.lower())
        logger.info(f"Original tokens: {tokens}")
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        logger.info(f"Processed tokens: {tokens}")
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return text

def train_model(data):
    """
    Train a Naive Bayes model on the provided data.

    Args:
        data (pd.DataFrame): The training data.

    Returns:
        model: The trained model.
    """
    try:
        data['processed_query'] = data['query'].apply(preprocess_text)
        X_train, X_test, y_train, y_test = train_test_split(data['processed_query'], data['intent'], test_size=0.2, random_state=42)

        # Log intent distribution
        logger.info(f"Training intents: {y_train.value_counts().to_dict()}")
        logger.info(f"Testing intents: {y_test.value_counts().to_dict()}")

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2))),  # Using unigrams and bigrams
            ('classifier', MultinomialNB())
        ])

        model = pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None

def predict_intent(model, query):
    """
    Predict the intent of a given query.

    Args:
        model: The trained model.
        query (str): The user's query.

    Returns:
        str: The predicted intent.
    """
    try:
        processed_query = preprocess_text(query)
        intent = model.predict([processed_query])[0]
        return intent
    except Exception as e:
        logger.error(f"Error predicting intent: {e}")
        return "unknown"

def respond_to_intent(intent):
    """
    Generate a response based on the predicted intent.

    Args:
        intent (str): The predicted intent.

    Returns:
        str: The bot's response.
    """
    responses = {
        "password_reset": "You can reset your password by clicking on 'Forgot Password' on the login page. If you need further assistance, please contact support.",
        "support_hours": "Our support hours are 9 AM to 5 PM, Monday to Friday.",
        "account_cancel": "To cancel your account, please visit the account settings page and follow the cancellation instructions.",
        "order_tracking": "You can track your order using the tracking link provided in your confirmation email.",
        "return_policy": "Our return policy allows you to return items within 30 days of receipt. Please visit our website for more details.",
        "subscription_change": "To change your subscription plan, log in to your account and navigate to the subscription settings.",
        "billing_update": "To update your billing information, go to your account settings and select 'Billing'.",
        "account_recovery": "If you've forgotten your account details, please visit our account recovery page.",
        "payment_methods": "We accept various payment methods, including credit/debit cards, PayPal, and bank transfers.",
        "refund_help": "For refund inquiries, please provide your order number and reason for the refund to our support team.",
        "report_problem": "If you encounter a problem with your order, please contact our support team with your order number.",
        "damaged_item": "If you receive a damaged item, please take a picture and contact our support team immediately.",
        "contact_support": "You can contact customer support via email at support@example.com or through our live chat feature.",
        "live_chat": "Yes, we offer live chat support during our business hours. Look for the chat icon on our website.",
        "product_documentation": "You can find product documentation in the 'Support' section of our website.",
        "shipping_address": "To change the shipping address on your order, please contact support as soon as possible.",
        "order_delay": "If your order is delayed, we will notify you via email with an updated delivery date.",
        "feedback": "We appreciate your feedback! Please let us know how we can improve our services.",
        "pause_subscription": "To pause your subscription, log in to your account and navigate to the subscription settings.",
        "missed_payment": "If you miss a payment, you will receive an email notification with further instructions.",
        "discounts": "Yes, we often have discounts available! Subscribe to our newsletter for the latest offers.",
        "delete_account": "To delete your account, please contact our support team with your request.",
        "speak_to_manager": "If you'd like to speak to a manager, please let our support team know your concerns.",
        "find_order_number": "You can find your order number in the confirmation email we sent you after your purchase.",
        "privacy_policy": "Our privacy policy outlines how we handle your personal information. You can view it on our website.",
        "newsletter_subscription": "You can subscribe to our newsletter by entering your email address on our website.",
        "gift_cards": "Yes, we offer gift cards! You can purchase them directly from our website.",
        "unsubscribe_emails": "To unsubscribe from our emails, please click the 'Unsubscribe' link at the bottom of any email.",
        "warranty_policy": "Our warranty policy covers all manufacturing defects for one year from the date of purchase.",
        "change_email": "To change your email address, please visit your account settings and update your contact information.",
        "unknown": "I'm sorry, I didn't understand that. Could you rephrase your question?"
    }
    return responses.get(intent, responses["unknown"])


def chatbot():
    """
    Main function to run the chatbot.
    """
    data = load_data()
    model = train_model(data)

    if model is None:
        logger.error("Model could not be trained. Exiting chatbot.")
        return

    print("Welcome to our Customer Support Chatbot! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Thank you for using the chatbot. Goodbye!")
            break
        intent = predict_intent(model, query)
        response = respond_to_intent(intent)
        print(f"Bot: {response}")
        logger.info(f"User query: {query} | Predicted intent: {intent} | Response: {response}")

if __name__ == "__main__":
    chatbot()
