def test_models_import():
    import models.schemas as schemas

    assert hasattr(schemas, "RAGResponse")
    assert hasattr(schemas, "QueryRequest")
