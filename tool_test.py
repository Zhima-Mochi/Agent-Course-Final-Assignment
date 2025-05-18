from app.infrastructure.tools_module import (
    wiki_search, web_search, arxiv_search, read_pdf, analyze_csv, analyze_excel,
    get_youtube_transcript, transcribe_youtube_audio, transcribe_audio, ocr_image, list_files,
    store_to_vectordb, vector_query
)

def test_wiki_search():
    print("Wiki Search Result:", wiki_search("Python (programming language)"))

def test_web_search():
    print("Web Search Result:", web_search("OpenAI"))

def test_arxiv_search():
    print("Arxiv Search Result:", arxiv_search("machine learning"))

def test_read_pdf():
    print("Read PDF Result:", read_pdf("sample.pdf"))  # Place a sample.pdf in task_files/

def test_analyze_csv():
    print("Analyze CSV Result:", analyze_csv("sample.csv"))  # Place a sample.csv in task_files/

def test_analyze_excel():
    print("Analyze Excel Result:", analyze_excel("sample.xlsx"))  # Place a sample.xlsx in task_files/

def test_get_youtube_transcript():
    print("YouTube Transcript Result:", get_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

def test_transcribe_youtube_audio():
    print("YouTube Audio Transcription Result:", transcribe_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

def test_transcribe_audio():
    print("Transcribe Audio Result:", transcribe_audio("sample.wav"))  # Place a sample.wav in task_files/

def test_ocr_image():
    print("OCR Image Result:", ocr_image("sample.png"))  # Place a sample.png in task_files/

def test_list_files():
    print("List Files Result:", list_files())

def test_store_to_vectordb():
    texts = ["What is AI?", "What is ML?"]
    metadatas = [{"answer": "Artificial Intelligence"}, {"answer": "Machine Learning"}]
    print("Store to VectorDB Result:", store_to_vectordb(texts, metadatas))

def test_vector_query():
    print("Vector Query Result:", vector_query("What is AI?"))

if __name__ == "__main__":
    test_wiki_search()
    # test_web_search()
    # test_arxiv_search()
    # test_read_pdf()
    # test_analyze_csv()
    # test_analyze_excel()
    # test_get_youtube_transcript()
    # test_transcribe_youtube_audio()
    # test_transcribe_audio()
    # test_ocr_image()
    # test_list_files()
    # test_store_to_vectordb()
    # test_vector_query()
