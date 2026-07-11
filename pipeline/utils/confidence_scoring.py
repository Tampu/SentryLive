# NOTE: this module is currently obsolete for the PoC
# token-level log probability scoring is unsupported by Gemini 2.5 Flash
# the factual and temporal verifier agents now serve as the primary confidence signals
# this module is retained for potential future use if a model supporting log probs is adopted
# --------------------------------------------------------------------------------------------- #



import math                 # used to convert raw log probabilities into a 0-1 confidence score for human readability
from typing import List     # used to enforce that function parameters expecting lists are typed correctly




# takes a list of token log probabilities and returns a single 0-1 confidence score
def compute_token_confidence(logprobs: List[float]) -> float:

    probs = [math.exp(lp) for lp in logprobs]   # loops through every returned log probability & converts to 0-1 scale
    return sum(probs) / len(probs)              # averages all probabilities into a single number & returns



# combines model's token-level confidence & retrieval strength into a single confidence report
def compute_confidence_score(logprobs: List[float], retrieval_score: float) -> dict:

    # get the 0-1 confidence score from the model's token probabilities
    token_confidence = compute_token_confidence(logprobs)

    # averages model's retrieval & confidence scores (0-1 scale)
    # - retrieval score   =  the model found the correct/relevant passages
    # - confidence score  =  model was sure about what it generated
    combined_score = (token_confidence + retrieval_score) / 2

    return {
        "token_confidence": round(token_confidence, 3),     # how confident the model was in its generated tokens
        "retrieval_score": round(retrieval_score, 3),       # how relevant the retrieved passages were to the query
        "combined_score": round(combined_score, 3),         # overall confidence — this is what gets shown in the demo
    }