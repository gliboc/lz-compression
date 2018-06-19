"""Lempel-Ziv 78 implementation"""

def compress(word):
    """Compression of a word using LZ78.

    Args:
    word (digit list): The word to compress.

    Returns:
    string list: The list of phrases used in LZ78.
    """

    phrases = set()
    current_prefix = ""
    nb_phrases = 0

    for digit in word:
        digit = str(digit)

        if current_prefix + digit in phrases:
            current_prefix += digit

        else:
            phrases.add(current_prefix + digit)
            nb_phrases += 1
            current_prefix = ""

    if current_prefix != "":
        #print("The last phrase is incomplete:", current_prefix)
        nb_phrases += 1

    #print("Compressed word")
    #input(phrases)
    return nb_phrases


def compress2(word):
    """Compress words as strings"""
    phrases = set()
    current_prefix = ""
    nb_phrases = 0
    for digit in word:

        if current_prefix + digit in phrases:
            current_prefix += digit

        else:
            phrases.add(current_prefix + digit)
            nb_phrases += 1
            current_prefix = ""

    if current_prefix != "":
        nb_phrases += 1

    return nb_phrases
    