from langchain_community.llms.ctransformers import CTransformers
# from langchain.chains.llm import  LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain_community.vectorstores import FAISS

from flask import Flask, render_template, request


# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

def load_llm(model_file):
    llm = CTransformers(
        model = model_file,#Tham số model của lớp CTransformers được thiết lập bằng giá trị của tham số model_file được truyền vào hàm. Điều này đặt mô hình sẽ được tải từ đường dẫn mà model_file trỏ tới.
        model_type = "llama",# loại mô hình mà chúng ta đang tải là loại "llama". Có thể đây là một loại kiến trúc hoặc tên gọi đặc biệt mà hàm CTransformers hỗ trợ.
        max_new_tokens = 4096, # là số lượng tokens tối đa mà mô hình có thể tạo ra khi được sử dụng để sinh văn bản mới.
        temperature = 0.3 #Temperature là một siêu tham số trong các mô hình sinh văn bản như GPT, nó ảnh hưởng đến sự đa dạng của văn bản được sinh ra. Giá trị thấp của temperature có thể dẫn đến việc tạo ra văn bản đa dạng hơn, nếu giá trị của temperature cao thì dẫn đến việc tạo ra văn bản ít đa dạng hơn
    )
    return llm


# Tạo prompts template để truyền cho model
def create_prompt(template):
    prompt = PromptTemplate(template= template, input_variables=["context","question"]) # context là những cái văn bản query trong vector db xong LLM sẽ dựa vào context và câu hỏi user để sinh ra câu trả lời
    return prompt


# Tạo simplechain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff", #stuff là một kiểu dữ liệu do người dùng tự định nghĩa, một cách tự do không bị ràng buộc bởi các kiểu dữ liệu tiêu chuẩn của ngôn ngữ.
        retriever = db.as_retriever(search_kwargs = {"k":1}),
        #kwargs cho phép bạn có thể thay đổi số liệu dựa trên ngữ cảnh cho phép bạn truyền vào hàm một số lượng bất kỳ các đối số và giá trị, và sau đó bạn có thể xử lý hoặc sử dụng chúng dựa trên ngữ cảnh cụ thể của hàm hoặc phương thức.
        # search_kwargs = {"k":3} là search là đưa ra n văn bản gần và giống nhất cụ thể ở đây là 3 văn bản gần và giống nhất
        return_source_documents = False,# khong cần trả lời thuộc văn bản nào, nếu muốn nó trả lời thuộc văn bản nào thì để True
        chain_type_kwargs= {'prompt': prompt}
    )
    return llm_chain

#Read từ vector DB
def read_vectors_db():
    #Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


#Chạy thử
# db = read_vectors_db()
# llm = load_llm(model_file)

# #Tạo prompt
# template = """system\n
# Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
# {context}
# user\n
# {question}\n
# assistant"""

# prompt = create_prompt(template)
# llm_chain = create_qa_chain(prompt, llm, db)


#Chạy chain
# question="Chế độ đau ốm của công ty canawan global  ?"
# response = llm_chain.invoke({"query": question})# invoke là hàm hỗ chợ các chức năng để chạy đối tượng llm_chain
# print(response)



# def main():
#     db = read_vectors_db()
#     llm = load_llm(model_file)
    
#     template = """system\n
#     Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
#     {context}
#     user\n
#     {question}\n
#     assistant"""

#     prompt = create_prompt(template)
#     llm_chain = create_qa_chain(prompt, llm, db)

#     while True:
#         question = input("Nhập câu hỏi của bạn: ")
#         if question.lower() == 'quit':
#             break
#         response = llm_chain.invoke({"query": question})
#         print("Câu trả lời:", response)

# if __name__ == "__main__":
#     main()




# Các hàm và import khác ở đây
app = Flask(__name__)

# Route cho trang chính
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        response = get_response(question)
        return render_template('index.html', question=question, response=response)
    return render_template('index.html')


# Hàm xử lý câu hỏi và nhận câu trả lời
def get_response(question):
    db = read_vectors_db()
    llm = load_llm(model_file)

    template = """system\n
    Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}
    user\n
    {question}\n
    assistant"""

    prompt = create_prompt(template)
    llm_chain = create_qa_chain(prompt, llm, db)


    response = llm_chain.invoke({"query": question})
    return response

if __name__ == "__main__":
    app.run(debug=True)