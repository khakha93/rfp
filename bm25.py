from fewshot_rerank import get_retriever
from tools import pretty_print_docs



if __name__ == "__main__":
    query = "기초학력 진단평가"

    esb_retriever = get_retriever()

    results = esb_retriever.invoke(query)
    pretty_print_docs(results[:])