# hnsw

Package `hnsw` implements Hierarchical Navigable Small World graphs in Go. You
can read up about how they work [here](https://arxiv.org/pdf/1603.09320). In essence,
they allow for fast approximate nearest neighbor searches with high-dimensional
vector data.

They are well-suited for semantic search applications on OpenAI embeddings.

## Roadmap

- [ ] Implement durable, file-system backend