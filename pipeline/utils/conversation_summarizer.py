import os                                                               # used to access GOOGLE_API_KEY from .env
from google import genai                                                # Google Gen AI client — used to summarize conversation history
from dotenv import load_dotenv                                          # loads environment variables from .env into os.getenv()
from pipeline.utils.logging_config import setup_logger                  # imports our custom logger for consistent logging across the pipeline



load_dotenv()                                                       # loads .env file so os.getenv() can access the API key

logger = setup_logger()                                             # initialize the logger for the conversation summarizer
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))          # initialize the Gemini client using the API key from .env




# summarizes the conversation history into a compact paragraph
# called every 3 turns to prevent context window overflow in long sessions
# keeps the context window manageable without losing important clinical context
def summarize_conversation(conversation_history: list, role: str = "clinician") -> str:

    logger.info(f"Conversation summarizer — summarizing {len(conversation_history)} turns")


    # build a readable transcript from the conversation history
    # formatted as Question/Answer pairs so Gemini can clearly distinguish questions from answers
    transcript = ""
    for turn in conversation_history:
        transcript += f"Q: {turn['query']}\n"

        # limit each answer to 300 characters — full answers can be very long
        # we only need enough for Gemini to understand what was discussed, not the full detail
        transcript += f"A: {turn['answer'][:300]}\n\n"



    # tailor the summary focus based on the user's role
    if role == "clinician":
        focus = "clinical questions, conditions, treatments, biomarkers, and key recommendations"
    else:
        focus = "what the patient asked, what they learned about their condition, and any guidance provided"




    # ask Gemini to produce a concise clinical summary of the conversation so far
    # the summary replaces the full history in the context window
    # focusing on clinical content ensures the summary is useful for future turns
    prompt = f"""You are summarizing a clinical conversation between a {role} and an oncology assistant.

Conversation transcript:
{transcript}

Summarize this conversation in 3-5 sentences. Focus on:
- {focus}

Keep the summary concise and relevant. Do not add new information."""




    # single Gemini call to generate the summary
    response = client.models.generate_content(
        model=os.getenv("ANSWER_GENERATION_MODEL", "gemini-2.5-flash"),
        contents=prompt,
    )


    summary = response.text.strip()     # extract and clean the summary text

    logger.info("Conversation summarizer — summary complete")

    return summary                      # returns a compact paragraph replacing the full history