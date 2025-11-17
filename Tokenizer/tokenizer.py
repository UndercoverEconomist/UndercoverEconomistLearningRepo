from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocab = None              # word -> frequency (BPE-ized)
        self.merges = []               # list of merged pairs in order
        self.merges_ranks = {}         # pair -> rank (index in merges)
        self.token2id = {}             # token -> int id
        self.id2token = {}             # int id -> token
        self.special_tokens = ["<pad>", "<unk>"]

    # ---------- Helpers for training ----------

    def _get_vocab(self, corpus):
        """
        Build word vocabulary as sequences of characters + </w>, with frequencies.
        corpus: list of strings (documents)
        """
        vocab = Counter()
        for line in corpus:
            for word in line.strip().split():
                if not word:
                    continue
                symbols = list(word) + ["</w>"]
                vocab[" ".join(symbols)] += 1
        return vocab

    def _get_stats(self, vocab):
        """
        Count frequencies of symbol pairs over the current vocab.
        vocab: dict word_str -> frequency, where word_str = "s y m b o l s"
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        """
        Merge all occurrences of 'pair' into a single symbol in the vocab.
        pair: tuple (sym1, sym2)
        """
        bigram = " ".join(pair)
        replacement = "".join(pair)
        new_vocab = Counter()
        for word, freq in vocab.items():
            # This works because symbols are space-separated
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    # ---------- Public training API ----------

    def train(self, corpus, vocab_size=1000, min_freq=2, verbose=False):
        """
        Train BPE tokenizer on a list of documents (strings).

        Args:
            corpus: list[str]  – training documents
            vocab_size: int    – max number of tokens (approx)
            min_freq: int      – stop if best pair appears < min_freq
            verbose: bool      – print progress
        """
        vocab = self._get_vocab(corpus)
        if verbose:
            print(f"Initial vocab size (unique words): {len(vocab)}")

        merges = []

        while True:
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            # Most frequent pair
            best_pair, best_freq = max(pairs.items(), key=lambda x: x[1])
            if best_freq < min_freq:
                if verbose:
                    print("Stopping: best pair frequency below min_freq")
                break

            # Rough approximation of symbol count to enforce vocab_size
            symbols = set()
            for word in vocab.keys():
                symbols.update(word.split())
            approx_symbol_count = len(symbols) + len(merges)
            if approx_symbol_count >= vocab_size:
                if verbose:
                    print("Stopping: reached vocab_size limit")
                break

            # Merge
            vocab = self._merge_vocab(best_pair, vocab)
            merges.append(best_pair)

            if verbose and len(merges) % 10 == 0:
                print(f"Merge {len(merges)}: {best_pair} (freq={best_freq})")

        # Save learned state
        self.vocab = vocab
        self.merges = merges
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}

        # Build final token list
        tokens = set(self.special_tokens)
        for word in vocab.keys():
            tokens.update(word.split())

        # stable ordering: special tokens first, then sorted rest
        non_special = sorted(t for t in tokens if t not in self.special_tokens)
        all_tokens = self.special_tokens + non_special

        self.token2id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        if verbose:
            print(f"Trained with {len(self.merges)} merges")
            print(f"Final vocab size (tokens): {len(all_tokens)}")

    # ---------- Encoding helpers ----------

    def _encode_word(self, word):
        """
        Encode a single word into BPE tokens using learned merges.
        """
        # Start from characters + word boundary marker
        symbols = list(word) + ["</w>"]
        if not self.merges_ranks:
            return symbols

        while True:
            # find all adjacent pairs
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

            # pick the pair with the lowest merge rank
            candidate = None
            candidate_rank = None
            for p in pairs:
                rank = self.merges_ranks.get(p)
                if rank is None:
                    continue
                if candidate is None or rank < candidate_rank:
                    candidate = p
                    candidate_rank = rank

            if candidate is None:
                # no more merge-able pairs
                break

            new_symbols = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and (symbols[i], symbols[i + 1]) == candidate
                ):
                    new_symbols.append("".join(candidate))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    # ---------- Public encode/decode API ----------

    def encode(self, text, return_ids=False):
        """
        Encode a string into BPE tokens or token IDs.

        Args:
            text: str
            return_ids: bool – if True, return list[int], else list[str]
        """
        tokens = []
        for word in text.strip().split():
            tokens.extend(self._encode_word(word))

        if return_ids:
            ids = [self.token2id.get(tok, self.token2id["<unk>"]) for tok in tokens]
            return ids
        return tokens

    def decode(self, ids_or_tokens):
        """
        Decode from token IDs or tokens back to a string.
        """
        if not ids_or_tokens:
            return ""

        if isinstance(ids_or_tokens[0], int):
            tokens = [self.id2token.get(i, "<unk>") for i in ids_or_tokens]
        else:
            tokens = ids_or_tokens

        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()
