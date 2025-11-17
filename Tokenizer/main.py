from tokenizer import BPETokenizer

if __name__ == "__main__":
    corpus = [
        "low lower lowest",
        "newer wider",
        "low low"
    ]

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=50, min_freq=1, verbose=True)

    # Encode to tokens
    tokens = tokenizer.encode("low lowest")
    print("Tokens:", tokens)

    # Encode to IDs
    ids = tokenizer.encode("low lowest", return_ids=True)
    print("IDs:", ids)

    # Decode back
    text = tokenizer.decode(ids)
    print("Decoded:", text)