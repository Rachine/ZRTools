"""Corpus specific functions should be defined here"""


def get_speaker(filename):
    """Get unique speaker ID from the name of the file
    """
    return filename.split('-')[0]
