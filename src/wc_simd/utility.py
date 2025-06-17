import json
import os


def spark_path(relative_path: str):
    # Compute the absolute path of your file
    absolute_path = os.path.abspath(relative_path)

    # Prepend with the "file://" scheme
    # Note: use three slashes (file:///)
    local_path = "file://" + absolute_path

    return local_path


def format_text(text, line_length=80):
    '''
    Formats the input text to fit within a specified line length.
    Args:
        text (str): The input text to format.
        line_length (int): The maximum length of each line.
    Returns:
        str: The formatted text with lines wrapped to the specified length.
    '''
    if text is None:
        return None
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def print_spark_dataframe_first_as_dict(df):
    print(json.dumps(df.first().asDict(), indent=4))
