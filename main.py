# entry point for the SentryLive pipeline
# run this file from the terminal to start a clinical QA chat session
# usage: python main.py

# ------------------------------------------------------------------------------------ #


from pipeline.orchestrator import run_pipeline          # imports the main pipeline function from the orchestrator


if __name__ == "__main__":

    print("\n" + "="*60)
    print("Welcome to SentryLive — Clinical QA Pipeline")
    print("CoRAL Lab, Arizona State University")
    print("="*60)


    # ask the user to declare their role at the start of the session
    # role determines which prompt template is used in Step 5
    print("\nPlease select your role:")
    print("  1. Clinician")
    print("  2. Patient")
    role_input = input("\nEnter 1 or 2: ").strip()
    role = "clinician" if role_input == "1" else "patient"
    print(f"\nRole set to: {role.capitalize()}")
    print("\nType your question and press Enter. Type 'exit' to quit.\n")


    # stores each turn of the conversation as a dict with 'query' and 'answer' keys
    # passed to the orchestrator on every run so the pipeline has access to prior context
    conversation_history = []



    # main chat loop — keeps running until the user types 'exit'
    while True:

        # get the user's query
        query = input("You: ").strip()

        # exit condition
        if query.lower() == "exit":
            print("\nEnding session. Goodbye!")
            break

        # skip empty inputs — prevents the pipeline from running on accidental blank enters
        if not query:
            continue

        # run the pipeline with the current query, role, and full conversation history
        result = run_pipeline(query=query, role=role, conversation_history=conversation_history)

        # display the answer
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(result["answer"])


        # display the verification report if available
        if result.get("verification"):
            print("=" * 60)
            print("VERIFICATION REPORT")
            print("=" * 60)
            print(result["verification"]["verification_text"])
            print("-" * 60 + "\n")


        # display the temporal verification report if available
        if result.get("temporal_verification"):
            print("=" * 60)
            print("TEMPORAL VERIFICATION REPORT")
            print("=" * 60)
            print(result["temporal_verification"]["verification_text"])
            print("-" * 60 + "\n")


        # store this turn in conversation history so future queries have access to prior context
        conversation_history.append({
            "query": query,             # the user's question
            "answer": result["answer"]  # the pipeline's response
        })


        # Step 7 — prompt clinicians for feedback after each answer
        # only clinicians are prompted — patients don't submit corrections
        # pressing Enter without typing skips feedback collection for that turn
        if role == "clinician":
            print("\nWas this answer correct? (press Enter to skip, or type your correction below)")
            correction = input("Correction: ").strip()

            # only store feedback if the clinician actually typed a correction
            # empty input means the answer was acceptable — no correction needed
            if correction:
                from pipeline.steps.step7_feedback_storage import store_feedback    # imported here to keep main.py lightweight
                store_feedback(
                    query=query,                                                    # the original clinical question
                    original_answer=result["answer"],                               # what the pipeline generated
                    correction=correction                                           # what the clinician says was wrong or missing
                )
                print("Correction saved. It will be applied to future similar queries.\n")