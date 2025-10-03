import re

URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)

SPAM_TOKENS = {
    "free","winner","win","prize","bonus","cash","loan","credit","offer","deal","discount",
    "limited","urgent","immediate","verify","account","lottery","weight","miracle","rich",
    "click","claim","now","act","exclusive","vip","guarantee","investment","bitcoin","crypto"
}

class EmailFeaturizer:
    """
    Extract the SAME feature names as in the provided CSV:
      - words            : total token count
      - links            : number of http(s) links
      - capital_words    : number of ALL-CAPS tokens (length >= 2)
      - spam_word_count  : number of tokens found in SPAM_TOKENS
    """
    def from_text(self, text: str):
        t = text or ""
        tokens = re.findall(r'\b\w+\b', t)
        words = len(tokens)
        links = len(URL_RE.findall(t))
        capital_words = sum(1 for tok in tokens if tok.isalpha() and tok.upper() == tok and len(tok) >= 2)
        tl = [tok.lower() for tok in tokens]
        spam_word_count = sum(1 for tok in tl if tok in SPAM_TOKENS)
        return {
            "words": float(words),
            "links": float(links),
            "capital_words": float(capital_words),
            "spam_word_count": float(spam_word_count),
        }
