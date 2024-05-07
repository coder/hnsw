# hnsw

Package `hnsw` implements Hierarchical Navigable Small World graphs in Go. You
can read up about how they work [here](https://arxiv.org/pdf/1603.09320). In essence,
they allow for fast approximate nearest neighbor searches with high-dimensional
vector data.

They are well-suited for semantic search applications on OpenAI embeddings.

This package can be thought of as an in-memory alternative to your favorite 
vector database (e.g. Pinecone, Weaviate). Granted, it implements just the essential
operations:

| Operation | Complexity            | Description                                  |
| --------- | --------------------- | -------------------------------------------- |
| Insert    | $O(log(n))$           | Insert a vector into the graph               |
| Delete    | $O(M^2 \cdot log(n))$ | Delete a vector from the graph               |
| Search    | $O(log(n))$           | Search for the nearest neighbors of a vector |
| Lookup    | $O(1)$                | Retrieve an object by ID                     |

> **Note**: Complexities are approximate where $n$ is the number of vectors in the graph
> and $M$ is the maximum number of neighbors each node can have. This [paper](https://arxiv.org/pdf/1603.09320) is a good resource for understanding the effect of
> the various construction parameters.

## Performance

By and large the greatest effect you can have on the performance of the graph
is reducing the dimensionality of your data. At 1536 dimensions (OpenAI default),
70% of the query process under default parameters is spent in the distance function. 

## Roadmap

- [ ] Implement durable, file-system backend