import os                                                               # used to access GOOGLE_API_KEY and model settings from .env
from google import genai                                                # Google Gen AI client — used to send the final prompt to Gemini and get an answer
from google.genai import types                                          # used to pass generation config parameters like temperature
from dotenv import load_dotenv                                          # loads environment variables from .env into os.getenv()
from pipeline.utils.logging_config import setup_logger                  # imports our custom logger for consistent logging across the pipeline
from pipeline.utils.confidence_scoring import compute_confidence_score  # imports our confidence scoring function to score the generated answer



load_dotenv()                                                       # loads .env file so os.getenv() can access the API keys

logger = setup_logger()                                             # initialize the logger for this step
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))          # initialize the Gemini client using the API key from .env




# takes the routed prompt from Step 5 and generates a grounded clinical answer with a confidence score
# accepts conversation history to maintain context across multiple turns in a chat session
def generate_answer(prompt: str, retrieval_score: float = 0.0, conversation_history: list = None) -> dict:

    logger.info("Step 6 — generating answer from Gemini")


    # use an empty list if no history is passed — avoids Python's mutable default argument gotcha
    conversation_history = conversation_history or []


    # build a conversation context string from prior turns so Gemini knows what was discussed before
    # only included if there is prior history — empty on the first turn of a session
    history_str = ""
    if conversation_history:
        history_str = "Previous conversation:\n"
        for turn in conversation_history:
            history_str += f"User: {turn['query']}\n"
            history_str += f"Assistant: {turn['answer']}\n"
        history_str += "\n"


    # read temperature from .env — defaults to 0.1 if not set
    # lower temperature = more deterministic and consistent clinical responses
    temperature = float(os.getenv("TEMPERATURE", "0.1"))


    # sends the role-specific prompt to Gemini and gets an answer back
    # note: logprobs not supported on gemini-2.5-flash — confidence score will use retrieval score only
    response = client.models.generate_content(
        model=os.getenv("ANSWER_GENERATION_MODEL", "gemini-2.5-flash"),     # uses model from .env, defaults to gemini-2.5-flash
        contents=history_str + prompt,                                      # prepends conversation history to the prompt so Gemini has full context
        config=types.GenerateContentConfig(
            temperature=temperature                                         # controls response randomness — loaded from .env
        )
    )
    answer = response.text                                                  # extract the generated answer text from the response



    # loop through each generated token and collect its log probability
    # these raw numbers get passed to compute_confidence_score() to produce a 0-1 confidence score
    logprobs = []
    if response.candidates and response.candidates[0].logprobs_result:
        for token in response.candidates[0].logprobs_result.chosen_candidates:
            logprobs.append(token.log_probability)                          # collect the log probability for each generated token



    # compute the confidence score combining token-level confidence and retrieval score
    # if Gemini doesn't return log probabilities, return None for token scores rather than crashing
    confidence = compute_confidence_score(logprobs, retrieval_score) if logprobs else {
        "token_confidence": None,
        "retrieval_score": round(retrieval_score, 3),
        "combined_score": None
    }



    logger.info(f"Step 6 — answer generated, confidence: {confidence}")

    return {
        "answer": answer,           # the generated clinical response
        "confidence": confidence,   # confidence score breakdown — token confidence, retrieval score, and combined score
    }