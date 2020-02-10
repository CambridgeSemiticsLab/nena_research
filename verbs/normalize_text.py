"""
This module contains code to normalize
the plain text of NENA word nodes.
This should later be turned into a separate
TF feature.
"""

import re
import unicodedata

def normalize_nena(word, tf):
    """Strip accents and spaces from NENA text on a node.
    
    Args:
        word: a node number to get normalized text
    """
    accents = '\u0300|\u0301|\u0304|\u0306|\u0308|\u0303'
    norm = tf.F.text.v(word).replace(tf.F.end.v(word),'')
    norm = unicodedata.normalize('NFD', norm) # decompose for accent stripping
    norm = re.sub(accents, '', norm) # strip accents
    return unicodedata.normalize('NFC', norm)
