
class CredentialsNotFoundError(Exception):
    """ Covers scenarios where a credential file has not been found"""
    def __init(self, message = "Credentials not found"):
        self.message = message
        super().__init__(message)

