import os
from email.message import EmailMessage
import ssl
import smtplib

class CustomLogger:
    def __init__(self):
        self.name = ""

    def notify_when_done(self) : 
        """send an email when finished"""
        subj = "Onyxia run — done"
        body = "https://projet-datalab-axel-morin-135428-0.lab.groupe-genes.fr/lab?"
        em = EmailMessage()
        em["From"] = os.environ["EMAIL_FROM"]
        em["To"] = os.environ["EMAIL_TO"]
        em["Subject"] = subj
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp : 
            print(smtp.login(os.environ["EMAIL_FROM"], os.environ["EMAIL_FROM_PWD"]))
            print(smtp.sendmail(
                os.environ["EMAIL_FROM"],
                os.environ["EMAIL_TO"], 
                em.as_string())
            )

    def __str__(self) -> str:
        return (
            "Custom Logger object"
        )