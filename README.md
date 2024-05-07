# hnsw

Package `hnsw` implements Hierarchical Navigable Small World graphs in Go. You
can read up about how they work [here](https://arxiv.org/pdf/1603.09320). In essence,
they allow for fast approximate nearest neighbor searches with high-dimensional
vector data.

They are well-suited for semantic search applications on OpenAI embeddings.

## Performance

By and large the greatest effect you can have on the performance of the graph
is reducing the dimensionality of your data. At 1536 dimensions (OpenAI default),
70% of the query process under default parameters is spent in the distance function. 

## Roadmap

- [ ] Implement durable, file-system backend