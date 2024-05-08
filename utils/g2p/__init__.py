""" from https://github.com/keithito/tacotron """
from utils.g2p import cleaners, mergers
from tokenizers import Tokenizer
import json
import re

class PhonemeBpeTokenizer:

  def __init__(self, tokenizer_path = "./utils/g2p/bpe_643.json"):
    self.tokenizer = Tokenizer.from_file(tokenizer_path)
    self.tokenizer_path = tokenizer_path

    with open(tokenizer_path, 'r') as f:
      json_data = f.read()
    data = json.loads(json_data)
    self.vocab = data['model']['vocab']

  def tokenize(self, text, language, merge=True):

    # 1. convert text to phoneme
    phonemes = _clean_text(text, ['cje_cleaners'])
    # print('clean text: ', phonemes)

    # 2. replace blank space " " with "_"
    phonemes = phonemes.replace(" ", "_")

    # 3. tokenize phonemes
    phoneme_tokens = self.tokenizer.encode(phonemes).ids
    # print('encode: ', phoneme_tokens)

    # 4. merge phoneme based on language [optional]
    if merge:
      phoneme_tokens = _merge_phoneme_token(phoneme_tokens, self.vocab, language, ['cj_mergers'])

    # # 5. decode tokens [optional]
    # decoded_text = self.tokenizer.decode(phoneme_tokens)
    # print('decoded: ', decoded_text)

    # # if not len(phoneme_tokens):
    # #   raise ValueError("Empty text is given")

    return phonemes, phoneme_tokens

def _clean_text(text, cleaner_names):

  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)

  return text

def _merge_phoneme_token(tokens, vocab, language, merger_names):

  for name in merger_names:
    merger = getattr(mergers, name)
    if not merger:
      raise Exception('Unknown merger: %s' % name)
    tokens = merger(tokens, vocab, language)

  return tokens
