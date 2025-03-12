import os

def storage_options():
    return {
        'client_kwargs': {'endpoint_url': 'https://minio-simple.lab.groupe-genes.fr'},
        'key': os.environ["AWS_ACCESS_KEY_ID"],
        'secret': os.environ["AWS_SECRET_ACCESS_KEY"],
        'token': os.environ["AWS_SESSION_TOKEN"]
    }