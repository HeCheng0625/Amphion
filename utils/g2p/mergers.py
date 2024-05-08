import re
from utils.g2p.japanese import japanese_merge_phoneme
from utils.g2p.mandarin import chinese_merge_phoneme

def cj_mergers(tokens, vocab, language):
    if language == 'zh':
        tokens = chinese_merge_phoneme(tokens, vocab)
    elif language == 'ja':
        tokens = japanese_merge_phoneme(tokens, vocab)
    return tokens
