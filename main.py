import sys

import nltk
from nltk.tag import DefaultTagger
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer


# The text with which to replace nouns.
REPLACE_TEXT = 'egg salad'


def detokenize(tokens):
  return TreebankWordDetokenizer().detokenize(tokens)


def word_span_tokenize(text):
  return TreebankWordTokenizer().span_tokenize(text)


def sent_span_tokenize(text):
  return PunktSentenceTokenizer().span_tokenize(text)


def pos_tag(tokens):
  return nltk.pos_tag(tokens)


def make_blocks(text_stream):
  """Convert the text stream into blocks / paragraphs."""
  blocks = []

  full_block = True
  block = ''
  for line in text_stream:
    if full_block:
      if line.strip() == '':
        blocks.append(('FULL', block))
        block = line
        full_block = False
      else:
        block += line
    else:
      if line.strip() != '':
        blocks.append(('EMPTY', block))
        block = line
        full_block = True
      else:
        block += line

  return blocks


def convert_sentence(sentence):
  """Convert a sentence, replacing nouns with the replacement text."""
  word_tokens = word_tokenize(sentence.lower())
  word_span_tokens = list(word_span_tokenize(sentence))
  tagged_sentence = pos_tag(word_tokens)
  assert len(word_tokens) == len(word_span_tokens) == len(tagged_sentence)

  converted_sentence = sentence
  offset = 0
  for (word, tag), span in zip(tagged_sentence, word_span_tokens):
    if tag == 'NN' or tag == 'NNS':
      adjusted_span = (span[0] + offset, span[1] + offset)

      replace_text = assimilate(
        REPLACE_TEXT,
        converted_sentence[adjusted_span[0]:adjusted_span[1]],
        tag
      )
      converted_sentence = replace(
        converted_sentence,
        adjusted_span[0],
        adjusted_span[1],
        replace_text
      )

      offset += len(replace_text) - (span[1] - span[0])

  return converted_sentence


def convert_block(block):
  """Convert a block of text (one or more lines of text).

  for example:
  PART THE FIRST.

  or:
     It is an ancient Mariner,
     And he stoppeth one of three.
     "By thy long grey beard and glittering eye,
     Now wherefore stopp'st thou me?

  or:
    <single line of whitespace>

  into processed form, preserving line breaks and formatting.
  """
  sentence_tokens = sent_tokenize(block)
  sentence_span_tokens = list(sent_span_tokenize(block))
  assert len(sentence_tokens) == len(sentence_span_tokens)

  converted_block = block
  offset = 0
  for sentence, span in zip(sentence_tokens, sentence_span_tokens):
    adjusted_span = (span[0] + offset, span[1] + offset)

    converted_sentence = convert_sentence(sentence)
    converted_block = replace(
      converted_block,
      adjusted_span[0],
      adjusted_span[1],
      converted_sentence
    )

    offset += len(converted_sentence) - (span[1] - span[0])

  return converted_block


def replace(original, start, end, patch):
  return original[:start] + patch + original[end:]


def assimilate(text, context, tag):
  """Modify the text to match the style of context and the desired noun-type tag."""
  if tag == 'NNS':
    text += 's'
  if context.isupper():
    # If context is all upper, either return all upper or, if context is one
    # char, just return title case.
    if len(context) == 1:
      return text[0].upper() + text[1:]
    return text.upper()
  elif context.islower():
    return text.lower()
  else:
    # If not all upper or lower, probably title case.
    return text[0].upper() + text[1:]


def _main():
  if '--help' in sys.argv or len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} <file>')
    sys.exit(1)

  with open(sys.argv[1]) as text_stream:
    converted_text = ''

    blocks = make_blocks(text_stream)
    for ty, block in blocks:
      if ty == 'EMPTY':
        converted_text += block
      else:
        converted_block = convert_block(block)
        converted_text += converted_block

    print(converted_text)


if __name__ == '__main__':
  _main()

