from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"

loader = WebBaseLoader(url)
docs = loader.load()

if docs:
    print(docs[0].page_content)
else:
    print("No content was loaded from the URL.")