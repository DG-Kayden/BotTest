from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.llm import  LLMChain
from langchain.prompts import PromptTemplate

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"

def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01
    )
    return llm


# Tạo prompts template để truyền cho model
def create_prompts(template):
    prompt = PromptTemplate(template= template, input_variables=['question'])
    return prompt


# Tạo simplechain
def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


#Chạy thử chain
template ="""<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


prompt = create_prompts(template)
llm = load_llm(model_file)
llm_chain = create_simple_chain(prompt, llm)

question = "Truyện Kiều là tác phẩm văn học của tác giả nào ?"
response = llm_chain.invoke({"question": question})
print(response)