from dotenv import load_dotenv
import warnings
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from pydantic import BaseModel, Field

from llama_index.core.program import FunctionCallingProgram
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

# context images
pdf_path = "./docs"
pdf_documents = SimpleDirectoryReader(pdf_path).load_data()

class InvoiceInfo(BaseModel):
    """Data model for an Invoice."""
    invoice_date_of_issue: str = Field(description="The date the invoice was issued.")
    invoice_number: str = Field(description="The unique number assigned to the invoice.")
    customer_name: str = Field(description="The name of the customer. eg: Tom Johnson")
    customer_address: str = Field(description="The address of the customer. eg: 123 Main St, City, Country")

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

prompt_template_str = """
system: You are an expert in extracting data from invoices.
extract the information from the invoice {invoice_text} and store it into the provided data class. """

program = FunctionCallingProgram.from_defaults(
    output_cls=InvoiceInfo,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

output = program(
    invoice_text=pdf_documents, description="Data model for an invoice."
)
