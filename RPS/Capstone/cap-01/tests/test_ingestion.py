def test_loader_import():
    import ingestion.loader as loader

    assert hasattr(loader, "load_file")
    assert hasattr(loader, "chunk_text")


def test_chunking_empty():
    import ingestion.loader as loader
    assert loader.chunk_text("") == []
