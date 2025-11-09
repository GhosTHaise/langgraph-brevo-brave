import brevo_python
import os
from brevo_python.rest import ApiException
from langchain_core.tools import tool

configuration = brevo_python.Configuration()
configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

if "BREVO_API_KEY" not in os.environ:
    os.environ["BREVO_API_KEY"] = getpass.getpass("Enter your Brevo API key: ")
    
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """ Tool that sends an email using Brevo. 
        Args:
            recipient (str): The email address of the recipient.
            subject (str): The subject of the email.
            body (str): The body of the email as html content.
    """
    api_instance = brevo_python.TransactionalEmailsApi(brevo_python.ApiClient(configuration))
    
    sender = {"name":"Agent - IA","email":"ghostrex2@gmail.com"}
    html_content = f"{body}"
    to = [{"email": f"{recipient}"}]
    cc = [{"email": "ghostrex2@gmail.com"}]
    bcc = [{"email": "ghostrex2@gmail.com"}]
    send_smtp_email = brevo_python.SendSmtpEmail(
        to=to, 
        bcc=bcc, 
        cc=cc, 
        html_content=html_content, 
        sender=sender, 
        subject=f"{subject}"
    ) # SendSmtpEmail | Values to send a transactional email

    try:
        # Send a transactional email
        api_response = api_instance.send_transac_email(send_smtp_email)
        print(api_response)
    except ApiException as e:
        print("Exception when calling TransactionalEmailsApi->send_transac_email: %s\n" % e)