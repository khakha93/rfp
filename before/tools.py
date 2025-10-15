from langchain_core.documents import Document



# 문서 출력 도우미 함수
def pretty_print_docs(docs: list[Document]):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
