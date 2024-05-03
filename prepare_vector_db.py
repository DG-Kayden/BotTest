from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter # type: ignore # RecursiveCharacterTextSplitter chia văn bản, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # PyPDFLoader để load file PDF, DirectoryLoader scan toàn bộ data
from langchain_community.vectorstores import FAISS # faiss thường được sử dụng cho việc xử lý vector lớn và tìm kiếm vector gần nhất nhanh chóng trong không gian nhiều chiều.
from langchain_community.embeddings import GPT4AllEmbeddings 
# GPT4AllEmbeddings có thể được thiết kế để cung cấp khả năng nhúng (embedding) cho văn bản sử dụng một mô hình GPT-4, hoặc một biến thể của mô hình này. 
# Nhúng văn bản là quá trình biến đổi các từ hoặc câu thành các vector số học, giúp máy tính hiểu được ý nghĩa của chúng trong không gian vector.

# Khai báo biến
pdf_data_path ="data"
vector_db_path = "vectorstores/db_faiss"


# 1. Tạo ra vector DB tu 1 đoạn text
def create_db_from_text():
    text = """BMW S1000RR là biểu tượng của sự hoàn hảo kỹ thuật trong thế giới mô tô. Với thiết kế sắc sảo và đường nét cứng cáp, nó tỏa ra vẻ mạnh mẽ ngay từ cái nhìn đầu tiên. Mỗi chi tiết được chăm chút kỹ lưỡng, từ khung sườn nhẹ nhàng cho đến các yếu tố tăng hiệu suất như đầu xylanh được tối ưu hóa và hệ thống làm mát cải tiến. 
    Động cơ của S1000RR là trái tim mạnh mẽ của chiếc xe, với một động cơ 4 xy-lanh dòng chảy, sản sinh ra một lượng công suất ấn tượng và động lực khí thải hứng khởi. Hệ thống truyền động và hộp số được tinh chỉnh đặc biệt để cung cấp phản ứng nhanh nhạy và trải nghiệm lái mạnh mẽ. 
    Tuy nhiên, sức mạnh của S1000RR được kiểm soát một cách chính xác và an toàn nhờ vào các công nghệ tiên tiến như hệ thống kiểm soát chống bó cứng phanh (ABS), hệ thống kiểm soát hành trình (Traction Control) và các chế độ lái điều chỉnh. Điều này giúp người lái cảm thấy tự tin khi chinh phục mọi loại đường. 
    Hệ thống treo linh hoạt và hệ thống phanh Brembo cao cấp không chỉ tạo ra sự ổn định và khả năng phanh an toàn, mà còn tạo ra một trải nghiệm lái tuyệt vời trên mọi loại địa hình. 
    Tóm lại, BMW S1000RR không chỉ là một mẫu mô tô tốc độ, mà còn là biểu tượng của sức mạnh, hiệu suất và công nghệ tiên tiến, mang đến cho người lái một trải nghiệm đầy cảm hứng và đầy quyền lực trên đường."""

# chia nhỏ text
    text_splitter = CharacterTextSplitter(
        separator="\n",# cắt mỗi đoạn là mỗi cái xuống dòng
        chunk_size=500,# mỗi đoạn 500 kí tự
        chunk_overlap=50,# mỗi đoạn cắt chồng chéo tối đa cắt ra 50 kí tự
        length_function=len
    )

    chunks = text_splitter.split_text(text)# chia text thành những đoạn nhỏ khác nhau

#Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

# Dựa vào faiss vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db


# Đọc file PDF
def create_db_from_files():
    # Khai báo loader để đọc toàn bộ thư mục data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls= PyPDFLoader) # pdf_data_path đọc file pdf trong data, glob="*.pdf" đọc file có đuôi là pdf, loader_cls= PyPDFLoader load(học) file pdf
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

#Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db
    

create_db_from_files()