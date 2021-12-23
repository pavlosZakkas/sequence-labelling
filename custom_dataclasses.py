from dataclasses import dataclass
from typing import List

@dataclass
class WordInfo():
  word: str
  pos: str
  bio: str

  def to_tuple(self):
      return (self.word, self.pos, self.bio)

@dataclass
class Sentence():
    words: List[WordInfo]

@dataclass
class Document():
    sentences: List[Sentence]