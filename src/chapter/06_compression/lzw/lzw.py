import logging
logger = logging.getLogger("lzw")


def lzw_compress(uncompressed_data: str, initializeDictionaryFromASCII: bool = True) -> tuple[list[int], dict[str, int]]:
    """
    Lempel-Ziv-Welch compression algorithm. 

    Args:
        uncompressed_data (str): input string, uncompressed.
        initializeDictionaryFromASCII: Initialize default dictionary with 256 ASCII chars.
        If False dictionary will be initialized with each unique char in uncompressed_data. 
        If False, this dictionary will contain the initial characters in alphabetically order (convention)!
        Default: True

    Returns:
        tuple[list[int], dict[str, int]]: Tuple of:
          - list[int]: list of encoded data
          - dict[str, int]: encoding dictionary
    """
    logger.debug(f"lzw_encode: {uncompressed_data}")

    # build initial dictionary
    if (initializeDictionaryFromASCII):
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
    else:
        # Initialize from unique chars in data
        dictionary = {c: i for i, c in enumerate(
            sorted(set(uncompressed_data)))}
        dict_size = len(dictionary.items())
    logger.debug(f"  Initial compress dictionary: {dictionary}")

    result = []
    w = ""
    for c in uncompressed_data:
        wc = w + c
        logger.debug(f"   c: {c} | wc: {wc}")
        if wc in dictionary:
            logger.debug(f"     -> found {wc} in dict")
            w = wc
        else:
            # wc not in dict, so take w
            logger.debug(f"     -> Append {dictionary[w]}")
            result.append(dictionary[w])

            # ... but add wc to dictionary.
            logger.debug(f"     -> Add to dict {wc} at {dict_size}")
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])

    return (result, dictionary)


def lzw_decompress(compressed_data: list[int], initializeDictionaryFromASCII: bool = True, decode_dictionary: dict[str, int] = None) -> str:
    """
    Lempel-Ziv-Welch decompression algorithm.

    Args:
        compressed_data (list[int]): The compressed data in form of a list of ints.
        initializeDictionaryFromASCII (bool): Initialize default dictionary with 256 ASCII chars.
          This must match the setting when compressing data! Default: True
        decode_dictionary (dict[str, int]): if initializeDictionaryFromASCII == False, then supply the
          decoding dictionary. Note: This dictionary will be swapped in keys/values within the function.

    Raises:
        ValueError: When some value is not in dictionary.

    Returns:
        str: Decompressed string object.
    """
    logger.debug(f"lzw_decompress: {compressed_data}")

    if (initializeDictionaryFromASCII):
        # build initial dictionary, inverse to lzw_compress
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
    else:
        if (decode_dictionary == None):
            raise ("Supply a dictionary when initializeDictionaryFromASCII == false.")
        # Swap dictionary keys and values
        dictionary = {v: k for k, v in decode_dictionary.items()}
        dict_size = len(decode_dictionary.items())
    logger.debug(f"  Initial decompress dictionary: {dictionary}")

    logger.debug(f"   First data elem: {compressed_data[0]}")
    result = ""
    if (initializeDictionaryFromASCII):
        w = chr(compressed_data[0])
    else:
        w = dictionary[compressed_data[0]]
    result += w
    for k in compressed_data[1:]:
        logger.debug(f"   k: {k}")
        if k in dictionary:
            logger.debug(f"     -> found {k} in dict")
            # tmp-store pattern of k (next)
            entry = dictionary[k]
        elif k == dict_size:
            # not yet known patterns _must_ have the end of the dict as index
            # tmp-store pattern of w (last)
            logger.debug(f"     -> not found {k} in dict")
            entry = dictionary[w]
        else:
            raise ValueError(f'Bad compressed key: {k}')
        result += entry

        # Add pattern(w)+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry
    return result
