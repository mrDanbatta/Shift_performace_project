import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Constructs a detailed error message including the file name, line number, and error message.
    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module to extract error details.
        Returns:
        str: A detailed error message.
    """

    # extract traceback details
    _, _, exc_tb = error_detail.exc_info()

    # get file name and line number from traceback
    file_name = exc_tb.tb_frame.f_code.co_filename

    # create formatted error message
    error_message = f"Error occurred in file: {file_name} at line number: {exc_tb.tb_lineno} with error message: {str(error)}"

    # log the error message
    logging.error(error_message)

    return error_message

class MyException(Exception):
    """
    Custom exception class that extends the base Exception class.
    It includes additional details about the error such as file name and line number.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Initializes the MyException instance with a detailed error message.
        Args:
            error_message (Exception): The original exception object.
            error_detail (sys): The sys module to extract error details.
        """
        super().__init__(error_message_detail(error_message, error_detail))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the MyException instance.
        Returns:
            str: The detailed error message.
        """
        return self.error_message

# Example usage:
# try:
#     raise MyException("An error occurred", sys)
# except MyException as e:
#     print(e)