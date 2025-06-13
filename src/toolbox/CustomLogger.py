# IMPORTS ######################################################################
from email.message import EmailMessage
import ssl
import smtplib
from .secrets import EMAIL_FROM, EMAIL_TO, EMAIL_FROM_PWD, URL_ONYXIA
# SCRIPTS ######################################################################
class CustomLogger:
    def __init__(self):
        self.name = ""

    def notify_when_done(self, message : str = '') : 
        """send an email when finished"""
        subj = "Onyxia run — done"
        body = (f"{URL_ONYXIA}\n"
                f"{message}")
        em = EmailMessage()
        em["From"] = EMAIL_FROM
        em["To"] = EMAIL_TO
        em["Subject"] = subj
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp : 
            print(smtp.login(EMAIL_FROM,EMAIL_FROM_PWD))
            print(smtp.sendmail(EMAIL_FROM,EMAIL_TO, em.as_string()))

    def __str__(self) -> str:
        return (
            "Custom Logger object"
        )