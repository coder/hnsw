# hnsw
[![GoDoc](https://godoc.org/github.com/golang/gddo?status.svg)](https://godoc.org/github.com/coder/hnsw)


Package `hnsw` implements Hierarchical Navigable Small World graphs in Go. You
can read up about how they work [here](https://www.pinecone.io/learn/series/faiss/hnsw/). In essence,
they allow for fast approximate nearest neighbor searches with high-dimensional
vector data.

This package can be thought of as an in-memory alternative to your favorite 
vector database (e.g. Pinecone, Weaviate). It implements just the essential
operations:

| Operation | Complexity            | Description                                  |
| --------- | --------------------- | -------------------------------------------- |
| Insert    | $O(log(n))$           | Insert a vector into the graph               |
| Delete    | $O(M^2 \cdot log(n))$ | Delete a vector from the graph               |
| Search    | $O(log(n))$           | Search for the nearest neighbors of a vector |
| Lookup    | $O(1)$                | Retrieve a vector by ID                      |

> [!NOTE]
> Complexities are approximate where $n$ is the number of vectors in the graph
> and $M$ is the maximum number of neighbors each node can have. This [paper](https://arxiv.org/pdf/1603.09320) is a good resource for understanding the effect of
> the various construction parameters.

## Usage

```
go get github.com/coder/hnsw@main
```

```go
g := hnsw.NewGraph[hnsw.Vector]()
g.Add(
    hnsw.MakeVector("1", []float32{1, 1, 1}),
    hnsw.MakeVector("2", []float32{1, -1, 0.999}),
    hnsw.MakeVector("3", []float32{1, 0, -0.5}),
)

neighbors := g.Search(
    []float32{0.5, 0.5, 0.5},
    1,
)
fmt.Printf("best friend: %v\n", neighbors[0].Embedding())
// Output: best friend: [1 1 1]
```

## Performance

By and large the greatest effect you can have on the performance of the graph
is reducing the dimensionality of your data. At 1536 dimensions (OpenAI default),
70% of the query process under default parameters is spent in the distance function.

If you're struggling with slowness / latency, consider:
* Reducing dimensionality
* Increasing $M$

And, if you're struggling with excess memory usage, consider:
* Reducing $M$ a.k.a `Graph.M` (the maximum number of neighbors each node can have)
* Reducing $M_l$ a.k.a `Graph.Ml` (the level generation parameter)


## Roadmap

- [ ] [#3](https://github.com/coder/hnsw/issues/3) Persistence / serialization