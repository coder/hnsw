# hnsw
[![GoDoc](https://godoc.org/github.com/golang/gddo?status.svg)](https://pkg.go.dev/github.com/coder/hnsw@main?utm_source=godoc)


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



## Persistence

While all graph operations are in-memory, `hnsw` provides facilities for loading/saving from persistent storage.

For an `io.Reader`/`io.Writer` interface, use `Graph.Export` and `Graph.Import`.

If you're using a single file as the backend, hnsw provides a convenient `SavedGraph` type instead:

```go
path := "some.graph"
g1, err := LoadSavedGraph[hnsw.Vector](path)
if err != nil {
    panic(err)
}
// Insert some vectors
for i := 0; i < 128; i++ {
    g1.Add(MakeVector(strconv.Itoa(i), []float32{float32(i)}))
}

// Save to disk
err = g1.Save()
if err != nil {
    panic(err)
}

// Later...
// g2 is a copy of g1
g2, err := LoadSavedGraph[Vector](path)
if err != nil {
    panic(err)
}
```

See more:
* [Export](https://pkg.go.dev/github.com/coder/hnsw#Graph.Export)
* [Import](https://pkg.go.dev/github.com/coder/hnsw#Graph.Import)
* [SavedGraph](https://pkg.go.dev/github.com/coder/hnsw#SavedGraph)

We use a fast binary encoding for the graph, so you can expect to save/load
nearly at disk speed. On my M3 Macbook I get these benchmark results:

```
goos: darwin
goarch: arm64
pkg: github.com/coder/hnsw
BenchmarkGraph_Import-16            2733            369803 ns/op         228.65 MB/s      352041 B/op       9880 allocs/op
BenchmarkGraph_Export-16            6046            194441 ns/op        1076.65 MB/s      261854 B/op       3760 allocs/op
PASS
ok      github.com/coder/hnsw   2.530s
```

when saving/loading a graph of 100 vectors with 256 dimensions.

## Performance

By and large the greatest effect you can have on the performance of the graph
is reducing the dimensionality of your data. At 1536 dimensions (OpenAI default),
70% of the query process under default parameters is spent in the distance function.

If you're struggling with slowness / latency, consider:
* Reducing dimensionality
* Increasing $M$

And, if you're struggling with excess memory usage, consider:
* Reducing $M$ a.k.a `Graph.M` (the maximum number of neighbors each node can have)
* Reducing $m_L$ a.k.a `Graph.Ml` (the level generation parameter)

### Memory Overhead

The memory overhead of a graph looks like:

$$
graph\_mem = n \cdot log(n) \cdot size(id) \cdot M\\
base\_mem = n \cdot d \cdot 4 \\
total\_memory = graph\_mem + base\_mem
$$

where:
* $n$ is the number of vectors in the graph
* $size(id)$ is the average size of the ID in bytes
* $M$ is the maximum number of neighbors each node can have
* $d$ is the dimensionality of the vectors
* $graph\_mem$ is the memory used by the graph structure across all layers
* $base\_mem$ is the memory used by the vectors themselves in the base or 0th layer

You can infer that:
* Connectivity or $M$ is very expensive if IDs are large
* If $d\cdot4$ is far larger than $M \cdot size(id)$, you should expect linear memory usage spent on representing vector data
* If $d\cdot4$ is far smaller than $M \cdot size(id)$, you should expect $n \cdot log(n)$ memory usage spent on representing graph structure

In the example of a graph 256 dimensions, and $M = 16$, with 8 byte IDs, you would see that each vector takes:

* $256 \cdot 4 = 1024$ data bytes 
* $16 \cdot 8 = 128$ metadata bytes

and memory growth is mostly linear.
