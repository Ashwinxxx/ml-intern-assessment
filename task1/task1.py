
import re
import random
from collections import defaultdict, Counter

class N_GramLanguageModel:
    def __init__(self, n=3, unk_threshold=1, start_token="<s>", end_token="</s>", unk_token="<unk>"):
        self.n = n
        self.unk_threshold = unk_threshold
        self.S = start_token
        self.E = end_token
        self.UNK = unk_token
        self.counts = defaultdict(Counter)
        self.probs = {}
        self.vocab = set()
        self.word_counts = Counter()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9.!?\n' ]+", " ", text)
        return text

    def split_sentences(self, text):
        text = re.sub(r"([.!?])", r"\1\n", text)
        return [s.strip() for s in text.split("\n") if s.strip()]

    def tokenize_sentence(self, sentence):
        return re.findall(r"[a-z0-9']+", sentence)

    def _compute_probs(self):
        for history, next_words in self.counts.items():
            total = sum(next_words.values())
            self.probs[history] = {w: c / total for w, c in next_words.items()}

    def fit(self, text):
        text = self.clean_text(text)
        sentences = self.split_sentences(text)
        tokenized = []

        for s in sentences:
            tokens = self.tokenize_sentence(s)
            self.word_counts.update(tokens)
            tokenized.append(tokens)

        self.vocab = {w for w, c in self.word_counts.items() if c >= self.unk_threshold}
        self.vocab.update({self.S, self.E, self.UNK})

        for tokens in tokenized:
            tokens = [t if t in self.vocab else self.UNK for t in tokens]
            padded = [self.S] * (self.n - 1) + tokens + [self.E]

            for i in range(len(padded) - self.n + 1):
                history = tuple(padded[i:i + self.n - 1])
                next_word = padded[i + self.n - 1]
                self.counts[history][next_word] += 1

        self._compute_probs()

    def _sample_next_word(self, history):
        if history in self.probs:
            words = list(self.probs[history].keys())
            probs = list(self.probs[history].values())
            return random.choices(words, weights=probs, k=1)[0]

        total = sum(self.word_counts.values())
        unigram_words = [w for w in self.vocab if w not in (self.S, self.E)]
        weights = [self.word_counts.get(w, 0) / total for w in unigram_words]
        return random.choices(unigram_words, weights=weights, k=1)[0]

    def generate(self, max_len=40):
        history = tuple([self.S] * (self.n - 1))
        out = []

        for _ in range(max_len):
            nxt = self._sample_next_word(history)
            if nxt == self.E:
                break
            out.append(nxt)
            history = history[1:] + (nxt,)

        text = " ".join(out)
        if text:
            text = text[0].upper() + text[1:]
        return text


if __name__ == "__main__":
    with open("booktxt.txt", "r", encoding="utf-8") as f:
        text = f.read()

    model = N_GramLanguageModel(n=3, unk_threshold=1)
    model.fit(text)

    print("Generated:", model.generate(50))
