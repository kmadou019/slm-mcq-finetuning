import re

def starts_with_negation(sentence: str):
    negation_words = r"^(not|no|don't|doesn't|isn't|aren't|wasn't|weren't|won't|can't|couldn't|shouldn't|wouldn't|didn't|haven't|hasn't|hadn't|mustn't)\b"
    
    if re.match(negation_words, sentence.strip(), re.IGNORECASE):
        return True
    return False
