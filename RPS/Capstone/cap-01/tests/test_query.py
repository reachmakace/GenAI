import rag_pipeline.query as q


def test_build_prompt_empty():
    p = q.build_prompt("What is X?", [])
    assert "Answer the question" in p


class FakeCollection:
    def query(self, query_texts, n_results=4):
        return {
            "documents": [["Doc content A", "Doc content B"]],
            "metadatas": [[{"source": "s1"}, {"source": "s2"}]]
        }


def test_query_collection_with_fake():
    coll = FakeCollection()
    res = q.query_collection(coll, "Explain A", k=2)
    assert "prompt" in res
    assert isinstance(res["retrieved"], list)
    assert res["answer"].startswith("[LLM") or isinstance(res["answer"], str)
