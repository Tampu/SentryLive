import os                                                               # used to access environment variables and file paths
import json                                                             # used to read and write doc_ids.json where document IDs are stored
from pathlib import Path                                                # cleaner way to work with file paths — handles OS differences automatically
from dotenv import load_dotenv                                          # loads .env file so the PageIndex API key is accessible
from pipeline.steps.step2a_guideline_retrieval import upload_guideline  # imports the upload function from Step 2A to avoid rewriting it
from pipeline.utils.drift_detector import is_update_document            # detects if a guideline PDF is an update to a previous guideline
import re                                                               # used to extract the year from the guideline filename using a regex pattern



load_dotenv()                                   # loads .env file so os.getenv() can access the API keys


# folder and file paths for guideline storage
GUIDELINES_DIR = Path("data/guidelines")        # folder where guideline PDFs are stored locally
DOC_IDS_FILE = GUIDELINES_DIR / "doc_ids.json"  # JSON file where PageIndex document IDs are saved after each upload



# opens doc_ids.json and returns its contents as a dictionary
# if the file doesn't exist yet, returns an empty structure so the rest of the script doesn't crash
def load_doc_ids() -> dict:
    if DOC_IDS_FILE.exists():
        with open(DOC_IDS_FILE, "r") as f:
            return json.load(f)
    return {"guidelines": {}}



# writes the updated doc IDs dictionary back to doc_ids.json
# called after every upload so no IDs are lost if the script crashes midway
def save_doc_ids(doc_ids: dict):
    with open(DOC_IDS_FILE, "w") as f:
        json.dump(doc_ids, f, indent=4)         # indent=4 makes the JSON file human-readable





# main function — loops through all PDFs in data/guidelines/ and uploads any that haven't been uploaded yet
# skips PDFs that already have a doc ID in doc_ids.json so we never upload the same file twice
def upload_all_guidelines():

    doc_ids = load_doc_ids()                            # load any existing doc IDs so we know what's already been uploaded


    for pdf_path in GUIDELINES_DIR.rglob("*.pdf"):      # find all PDF files in the guidelines folder and subdirectories
        name = pdf_path.stem                            # get the filename without the .pdf extension — used as the key in doc_ids.json

        # skip this PDF if it has already been uploaded
        if name in doc_ids["guidelines"]:
            print(f"Skipping {name} — already uploaded")
            continue


        print(f"Uploading {name}...")
        doc_id = upload_guideline(str(pdf_path))        # upload the PDF to PageIndex and get the document ID back


        # extract the year from the filename using regex
        # looks for a 4-digit number starting with 19 or 20 (e.g. 2018, 2024)
        # returns None if no year is found in the filename so the upload doesn't crash
        year_match = re.search(r'\b(19|20)\d{2}\b', name)
        year = int(year_match.group()) if year_match else None


        # detect if this document is an update to a previous guideline
        is_update = is_update_document(str(pdf_path))   # scans first 3 pages for ASCO update language


        # store structured metadata for each guideline
        doc_ids["guidelines"][name] = {
            "doc_id": doc_id,                           # PageIndex document ID — used to retrieve passages from this guideline
            "year": year,                               # year extracted from filename — used to determine which guideline is newer
            "is_update": is_update,                     # True if this document updates a previous guideline — used for drift detection
        }

        save_doc_ids(doc_ids)                           # save immediately after each upload in case of interruption
        print(f"Uploaded {name} — doc_id: {doc_id} | year: {year}")


    print(f"\nAll guidelines uploaded. doc_ids saved to {DOC_IDS_FILE}")





# only runs when this script is executed directly — not when imported by another file
if __name__ == "__main__":
    upload_all_guidelines()