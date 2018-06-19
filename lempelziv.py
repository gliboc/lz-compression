def compress(word):
    """Compression of a word using LZ78.

    Args:
    word (digit list): The word to compress.

    Returns:
    string list: The list of phrases used in LZ78.
    """

    s = set()
    current_prefix = ""
    nb_phrases = 0

    for digit in word:
        digit = str(digit)

        if current_prefix + digit in s:
            current_prefix += digit

        else:
            s.add(current_prefix + digit)
            nb_phrases += 1
            current_prefix = ""

    if current_prefix != "":
        #print("The last phrase is incomplete:", current_prefix)
        nb_phrases += 1

    #print("Compressed word")
    #input(phrases)
    return nb_phrases


def compress2(w):
    """Compress words as strings"""
    s = set()
    current_prefix = ""
    nb_phrases = 0
    for digit in w:

        if current_prefix + digit in s:
            current_prefix += digit

        else:
            s.add(current_prefix + digit)
            nb_phrases += 1
            current_prefix = ""

    if current_prefix != "":
        nb_phrases += 1

    return nb_phrases