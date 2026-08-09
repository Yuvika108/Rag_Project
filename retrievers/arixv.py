from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_metadata=True,
)

docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"Document {i + 1}:")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)
