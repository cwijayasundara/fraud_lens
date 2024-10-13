from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.bridge.pydantic import BaseModel, Field

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

# context images
image_path = "./images/passport"
image_documents = SimpleDirectoryReader(image_path).load_data()
print("Image documents:", image_documents)

# Desired output structure
class PassportInfo(BaseModel):
    """Data class for storing text attributes of a passport."""
    passport_number: str = Field(description="Passport number.")
    passport_code: str = Field(description="Passport code.")
    passport_type: str = Field(description="Passport type.")
    surname: str = Field(description="Surname of the passport holder.")
    given_names: str = Field(description="Given names of the passport holder.")
    nationality: str = Field(description="Nationality of the passport holder.")
    date_of_birth: str = Field(description="Date of birth of the passport holder.")
    sex: str = Field(description="sex of the passport holder.")
    place_of_birth: str = Field(description="Place of birth of the passport holder.")
    date_of_issue: str = Field(description="Date of issue of the passport.")
    date_of_expiry: str = Field(description="Date of expiry of the passport.")

passport_image_extraction_prompt = """
    Use the attached passport image to extract data from it and store into the
    provided data class.
"""

gpt_4o = OpenAIMultiModal(model="gpt-4o-2024-05-13", max_new_tokens=4096)
# gpt_4o = OpenAIMultiModal(model="gpt-4o-2024-08-06", max_new_tokens=8000)

multimodal_llms = {
    "gpt_4o": gpt_4o,
}

programs = {
    mdl_name: MultiModalLLMCompletionProgram.from_defaults(
        output_cls=PassportInfo,
        prompt_template_str=passport_image_extraction_prompt,
        multi_modal_llm=mdl,
    )
    for mdl_name, mdl in multimodal_llms.items()
}

# Please ensure you're using llama-index-core v0.10.37
passport_info = programs["gpt_4o"](image_documents=[image_documents[0]])

print("Passport info:", passport_info)