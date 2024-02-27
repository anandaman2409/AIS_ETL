class CategoricalColumnNotFoundException(Exception):
    def __init__(self, message: str):
        self.message = message
        Exception.__init__(self, message)

    def __str__(self):
        return f"CategoricalColumnNotFoundException: {self.message}"


class CategoricalDataUnrecognizedException(Exception):
    def __init__(self, message: str):
        self.message = message
        Exception.__init__(self, message)

    def __str__(self):
        return f"CategoricalDataUnrecognizedException: {self.message}"


class PredictColumnNotFoundException(Exception):
    def __init__(self, message: str):
        self.message = message
        Exception.__init__(self, message)

    def __str__(self):
        return f"PredictColumnNotFoundException: {self.message}"


class InvalidInputException(Exception):
    def __init__(self, message: str):
        self.message = message
        Exception.__init__(self, message)

    def __str__(self):
        return f"InvalidInputException: {self.message}"