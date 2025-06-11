import os
from email.message import EmailMessage
import ssl
import smtplib

class CustomLogger:
    def __init__(self, text_filename : str = None):
        self.name = ""
        self.text_filename = text_filename

    def log(self, message, printing : bool = False):
        if printing:
            print(message)
        # Save the message in the log
        if self.text_filename is None:
            print("-- No log file available --")
        else: 
            try : 
                #log file already exists
                with open(self.text_filename, "a") as file:
                    file.write(message)
            except : 
                # log file does not exist
                with open(self.text_filename, "w") as file:
                    file.write(message)
                    
    def notify_when_done(self, message : str = '') : 
        """send an email when finished"""
        subj = "Onyxia run — done"
        body = ("https://projet-datalab-axel-morin-437675-0.lab.groupe-genes.fr/?folder=/home/onyxia/work"
                f"\n{message}")
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