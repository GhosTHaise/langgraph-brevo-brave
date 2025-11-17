"""tools.py"""
import os
import getpass
import brevo_python
from brevo_python.rest import ApiException
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

# Ensure API Keys are set
if "BREVO_API_KEY" not in os.environ:
    os.environ["BREVO_API_KEY"] = getpass.getpass("Enter your Brevo API key: ")

# Initialize Brevo
configuration = brevo_python.Configuration()
configuration.api_key['api-key'] = os.environ["BREVO_API_KEY"]

# Secondary model for generating content (can be same as main model)
# Using a lightweight model for generation to be fast
generator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

@tool
def generate_email_body(subject: str, context: str) -> str:
    """
    Generates a professional HTML email body based on the subject and context.
    Returns the raw HTML string.
    """
    print(f"--- Generating Email Body for: {subject} ---")
    
    prompt = (
        f"Write a professional HTML email body. \n"
        f"Subject: {subject}\n"
        f"Context/Details: {context}\n"
        "Requirements: return ONLY the HTML code. Do not include markdown (```html). "
        "Do not include the subject line inside the HTML body."
    )
    
    response = generator_llm.invoke([HumanMessage(content=prompt)])
    html_content = response.content.strip().replace("```html", "").replace("```", "")
    
    return html_content

@tool
def send_email(recipient: str, subject: str, html_content: str) -> str:
    """
    Sends an email using Brevo.
    Args:
        recipient (str): The email address of the recipient.
        subject (str): The subject of the email.
        html_content (str): The full HTML body content.
    """
    print(f"--- Sending Email to {recipient} ---")
    
    api_instance = brevo_python.TransactionalEmailsApi(brevo_python.ApiClient(configuration))
    
    # sender info - Replace with your verified sender
    sender = {"name": "Agent IA", "email": "ghostrex2@gmail.com"}
    to = [{"email": recipient}]
    
    send_smtp_email = brevo_python.SendSmtpEmail(
        to=to,
        html_content=html_content,
        sender=sender,
        subject=subject
    )

    try:
        api_instance.send_transac_email(send_smtp_email)
        return f"SUCCESS: Email sent to {recipient} with subject '{subject}'."
    except ApiException as e:
        return f"ERROR: Failed to send email. Reason: {e}"