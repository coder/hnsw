# hnsw

[![GoDoc](https://godoc.org/github.com/golang/gddo?status.svg)](https://pkg.go.dev/github.com/coder/hnsw@main?utm_source=godoc)
![Go workflow status](https://github.com/coder/hnsw/actions/workflows/go.yaml/badge.svg)

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

```text
go get github.com/TFMV/hnsw@main
```

### Basic Usage

```go
// Create a new graph with default parameters
g := hnsw.NewGraph[int]()

// Add some vectors
g.Add(
    hnsw.MakeNode(1, []float32{1, 1, 1}),
    hnsw.MakeNode(2, []float32{1, -1, 0.999}),
    hnsw.MakeNode(3, []float32{1, 0, -0.5}),
)

// Search for the nearest neighbor
neighbors, err := g.Search(
    []float32{0.5, 0.5, 0.5},
    1,
)
if err != nil {
    log.Fatalf("failed to search graph: %v", err)
}
fmt.Printf("best friend: %v\n", neighbors[0].Value)
// Output: best friend: [1 1 1]
```

### Thread-Safe Usage

```go
// Create a thread-safe graph with custom parameters
g, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, hnsw.EuclideanDistance)
if err != nil {
    log.Fatalf("failed to create graph: %v", err)
}

// Add some initial nodes
g.Add(
    hnsw.MakeNode(1, []float32{1, 1, 1}),
    hnsw.MakeNode(2, []float32{1, -1, 0.999}),
    hnsw.MakeNode(3, []float32{1, 0, -0.5}),
)

// Perform concurrent operations
var wg sync.WaitGroup
numOperations := 10

// Concurrent searches
wg.Add(numOperations)
for i := 0; i < numOperations; i++ {
    go func(i int) {
        defer wg.Done()
        query := []float32{float32(i) * 0.1, float32(i) * 0.1, float32(i) * 0.1}
        results, err := g.Search(query, 1)
        if err != nil {
            log.Printf("Search error: %v", err)
            return
        }
        fmt.Printf("Search %d found: %v\n", i, results[0].Key)
    }(i)
}

// Wait for all operations to complete
wg.Wait()
```

## Persistence

While all graph operations are in-memory, `hnsw` provides facilities for loading/saving from persistent storage.

For an `io.Reader`/`io.Writer` interface, use `Graph.Export` and `Graph.Import`.

If you're using a single file as the backend, hnsw provides a convenient `SavedGraph` type instead:

```go
path := "some.graph"
g1, err := LoadSavedGraph[int](path)
if err != nil {
    panic(err)
}
// Insert some vectors
for i := 0; i < 128; i++ {
    g1.Add(hnsw.MakeNode(i, []float32{float32(i)}))
}

// Save to disk
err = g1.Save()
if err != nil {
    panic(err)
}

// Later...
// g2 is a copy of g1
g2, err := LoadSavedGraph[int](path)
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

```text
goos: darwin
goarch: arm64
pkg: github.com/coder/hnsw
BenchmarkGraph_Import-16            4029            259927 ns/op         796.85 MB/s      496022 B/op       3212 allocs/op
BenchmarkGraph_Export-16            7042            168028 ns/op        1232.49 MB/s      239886 B/op       2388 allocs/op
PASS
ok      github.com/coder/hnsw   2.624s
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

## Memory Overhead

The memory overhead of a graph looks like:

$$
\displaylines{
mem_{graph} = n \cdot \log(n) \cdot \text{size(id)} \cdot M \\
mem_{base} = n \cdot d \cdot 4 \\
mem_{total} = mem_{graph} + mem_{base}
}
$$

where:

* $n$ is the number of vectors in the graph
* $\text{size(key)}$ is the average size of the key in bytes
* $M$ is the maximum number of neighbors each node can have
* $d$ is the dimensionality of the vectors
* $mem_{graph}$ is the memory used by the graph structure across all layers
* $mem_{base}$ is the memory used by the vectors themselves in the base or 0th layer

You can infer that:

* Connectivity ($M$) is very expensive if keys are large
* If $d \cdot 4$ is far larger than $M \cdot \text{size(key)}$, you should expect linear memory usage spent on representing vector data
* If $d \cdot 4$ is far smaller than $M \cdot \text{size(key)}$, you should expect $n \cdot \log(n)$ memory usage spent on representing graph structure

In the example of a graph with 256 dimensions, and $M = 16$, with 8 byte keys, you would see that each vector takes:

* $256 \cdot 4 = 1024$ data bytes
* $16 \cdot 8 = 128$ metadata bytes

and memory growth is mostly linear.

---

## Recent Changes (TFMV)

### Error Handling Improvements

* Comprehensive error checking for invalid parameters:
  * Validation for M, Ml, and EfSearch parameters
  * Dimension matching between query vectors and graph vectors
  * Proper handling of nil nodes and edge cases

### Experimental Performance Optimizations

* **Vectorized Distance Calculations**: Improved performance for high-dimensional vectors.
* **Parallel Search**: Added `ParallelSearch` method that automatically parallelizes search operations for large graphs and high-dimensional data.
* **Optimized Node Management**: Enhanced neighbor selection and connectivity maintenance.
* **Thread-Safe Implementation**: Added synchronization primitives to enable concurrent operations.
* **Batch Operations**: Added `BatchAdd`, `BatchSearch`, and `BatchDelete` methods for high-throughput scenarios.

### Experimental New APIs

* **Configuration Validation**: Added `Validate()` method to check graph configuration parameters.
* **NewGraphWithConfig**: Constructor that validates parameters during graph creation.
* **MakeNode**: Helper function to create nodes with proper typing.
* **Quality Metrics**: New `QualityMetrics()` method in the Analyzer to evaluate graph quality.
* **BatchAdd**: Add multiple nodes in a single operation with a single lock acquisition.
* **BatchSearch**: Perform multiple searches in a single operation with a single lock acquisition.
* **BatchDelete**: Remove multiple nodes in a single operation with a single lock acquisition, returning a boolean slice indicating which keys were successfully deleted.
* **Negative Examples**: New methods to search with negative examples:
  * `SearchWithNegative`: Search with a single negative example.
  * `SearchWithNegatives`: Search with multiple negative examples.
  * `BatchSearchWithNegatives`: Batch search with negative examples.

### Experimental Quality Metrics

The Analyzer now provides experimental quality metrics to evaluate graph structure:

* **Connectivity Analysis**: Measure average connections per node and distribution.
* **Distortion Ratio**: Evaluate how well the graph preserves actual distances.
* **Layer Balance**: Assess the distribution of nodes across layers.
* **Graph Structure**: Analyze topography and connectivity patterns.

Example usage:

```go
analyzer := hnsw.Analyzer[int]{Graph: graph}
metrics := analyzer.QualityMetrics()

fmt.Printf("Node count: %d\n", metrics.NodeCount)
fmt.Printf("Average connectivity: %.2f\n", metrics.AvgConnectivity)
fmt.Printf("Layer balance: %.2f\n", metrics.LayerBalance)
fmt.Printf("Distortion ratio: %.2f\n", metrics.DistortionRatio)
```

### Testing

* **Benchmark Suite**: Comprehensive benchmarks for various graph sizes and dimensions.
* **Validation Tests**: Tests to ensure proper error handling for invalid configurations.
* **Performance Comparisons**: Benchmarks comparing sequential vs. parallel search performance.
* **Quality Metrics Tests**: Validation of graph quality measurement functions.
* **Concurrency Tests**: Tests to verify thread safety under concurrent operations.

### Thread Safety

The HNSW implementation now supports concurrent operations through a thread-safe design:

```go
// Create a thread-safe graph
g, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, hnsw.EuclideanDistance)

// Concurrent operations are now safe
var wg sync.WaitGroup
numOperations := 10

// Concurrent searches
wg.Add(numOperations)
for i := 0; i < numOperations; i++ {
    go func(i int) {
        defer wg.Done()
        query := []float32{float32(i) * 0.1, float32(i) * 0.1, float32(i) * 0.1}
        results, _ := g.Search(query, 1)
        fmt.Printf("Search %d found: %v\n", i, results[0].Key)
    }(i)
}

// Concurrent adds
wg.Add(numOperations)
for i := 0; i < numOperations; i++ {
    go func(i int) {
        defer wg.Done()
        nodeID := 10 + i
        vector := []float32{float32(i), float32(i), float32(i)}
        g.Add(hnsw.MakeNode(nodeID, vector))
    }(i)
}

// Wait for all operations to complete
wg.Wait()
```

### Batch Operations

For high-throughput scenarios, batch operations can significantly reduce lock contention:

```go
// Add a batch of nodes in a single operation
batch := make([]hnsw.Node[int], 5)
for i := range batch {
    nodeID := 100 + i
    vector := []float32{float32(i) * 0.5, float32(i) * 0.5, float32(i) * 0.5}
    batch[i] = hnsw.MakeNode(nodeID, vector)
}
g.BatchAdd(batch)

// Perform multiple searches in a single operation
queries := [][]float32{
    {0.1, 0.1, 0.1},
    {0.2, 0.2, 0.2},
    {0.3, 0.3, 0.3},
}
batchResults, _ := g.BatchSearch(queries, 2)

// Delete multiple nodes in a single operation
keysToDelete := []int{100, 101, 102}
deleteResults := g.BatchDelete(keysToDelete)
for i, key := range keysToDelete {
    fmt.Printf("Deleted %d: %v\n", key, deleteResults[i])
}
```

### Negative Examples

The HNSW implementation now supports searching with negative examples, allowing you to find vectors similar to your query but dissimilar to specified negative examples:

```go
// Create a graph with some vectors
g, err := hnsw.NewGraphWithConfig[string](16, 0.25, 20, hnsw.CosineDistance)
if err != nil {
    log.Fatalf("failed to create graph: %v", err)
}

// Add some vectors representing different concepts
g.Add(
    hnsw.MakeNode("dog", []float32{1.0, 0.2, 0.1, 0.0}),
    hnsw.MakeNode("puppy", []float32{0.9, 0.3, 0.2, 0.1}),
    hnsw.MakeNode("cat", []float32{0.1, 1.0, 0.2, 0.0}),
    hnsw.MakeNode("kitten", []float32{0.2, 0.9, 0.3, 0.1}),
    // ... more vectors ...
)

// Search with a single negative example
dogQuery := []float32{1.0, 0.2, 0.1, 0.0}      // dog query
puppyNegative := []float32{0.9, 0.3, 0.2, 0.1} // puppy (negative example)

// Find dog-related concepts but not puppies (negativeWeight = 0.5)
results, err := g.SearchWithNegative(dogQuery, puppyNegative, 3, 0.5)
if err != nil {
    log.Fatalf("failed to search with negative: %v", err)
}

// Search with multiple negative examples
petQuery := []float32{0.3, 0.3, 0.3, 0.3}      // general pet query
dogNegative := []float32{1.0, 0.2, 0.1, 0.0}   // dog (negative example)
catNegative := []float32{0.1, 1.0, 0.2, 0.0}   // cat (negative example)

negatives := []hnsw.Vector{dogNegative, catNegative}

// Find pet-related concepts but not dogs or cats (negativeWeight = 0.7)
results, err = g.SearchWithNegatives(petQuery, negatives, 3, 0.7)
if err != nil {
    log.Fatalf("failed to search with negatives: %v", err)
}

// Batch search with negative examples
queries := []hnsw.Vector{
    {1.0, 0.2, 0.1, 0.0}, // dog query
    {0.1, 1.0, 0.2, 0.0}, // cat query
}

batchNegatives := [][]hnsw.Vector{
    {
        {0.9, 0.3, 0.2, 0.1}, // puppy (negative for dog query)
    },
    {
        {0.2, 0.9, 0.3, 0.1}, // kitten (negative for cat query)
    },
}

batchResults, err := g.BatchSearchWithNegatives(queries, batchNegatives, 3, 0.5)
if err != nil {
    log.Fatalf("failed to batch search with negatives: %v", err)
}
```

#### Negative Examples Design

The negative examples feature allows you to find vectors that are similar to your query but dissimilar to specified negative examples:

1. **SearchWithNegative**: Search with a single negative example.

   ```go
   func (g *Graph[K]) SearchWithNegative(query, negative Vector, k int, negativeWeight float32) ([]Node[K], error)
   ```

2. **SearchWithNegatives**: Search with multiple negative examples.

   ```go
   func (g *Graph[K]) SearchWithNegatives(query Vector, negatives []Vector, k int, negativeWeight float32) ([]Node[K], error)
   ```

3. **BatchSearchWithNegatives**: Perform multiple searches with negative examples in a single operation.

   ```go
   func (g *Graph[K]) BatchSearchWithNegatives(queries []Vector, negatives [][]Vector, k int, negativeWeight float32) ([][]Node[K], error)
   ```

4. **Parameters**:
   * **query**: The query vector to search for.
   * **negative/negatives**: The negative example vector(s) to avoid.
   * **k**: The number of results to return.
   * **negativeWeight**: The weight to apply to negative examples (0.0 to 1.0).
     * 0.0: Ignore negative examples completely.
     * 1.0: Strongly avoid negative examples.
     * Recommended range: 0.3 to 0.7 for balanced results.

5. **Use Cases**:
   * **Content Filtering**: Find content similar to a query but excluding specific categories.
   * **Diversity**: Ensure diverse search results by avoiding similar items.
   * **Preference Learning**: Incorporate user preferences by avoiding disliked items.
   * **Semantic Search**: Refine search results by excluding irrelevant concepts.

6. **Performance Considerations**:
   * The search process remains efficient with O(log n) complexity.
   * The scoring calculation includes additional distance computations for negative examples.
   * For many negative examples, consider using a smaller negativeWeight to maintain result quality.

7. **Implementation Details**:
   * Results are scored based on similarity to the query and dissimilarity to negative examples.
   * The final score is a weighted combination: `score = similarity - (negativeWeight * negativeSimilarity)`.
   * Results are sorted by this combined score, returning the top k items.

This feature enables more nuanced and refined search capabilities, allowing you to express not just what you're looking for, but also what you want to avoid.

### Benchmark Results

Our benchmarks show the performance characteristics of the thread-safe implementation:

#### Sequential vs. Concurrent Operations

| Operation | Sequential (ns/op) | Concurrent (ns/op) | Notes |
|-----------|-------------------|-------------------|-------|
| Add       | 4,967             | 9,425             | Concurrent adds are slower due to lock contention |
| Search    | 32,758            | 16,967            | Concurrent searches are faster due to parallelism |
| Delete    | 22,131            | 399.1             | Batch deletes are more efficient for large operations |

#### Batch vs. Individual Operations

| Operation | Batch (ns/op) | Individual (ns/op) | Notes |
|-----------|--------------|-------------------|-------|
| Add       | 18,291,067    | 18,134,883        | Comparable performance for large batches |
| Search    | 2,923,725     | 3,036,183         | Batch searches are slightly faster |
| Delete    | 22,131        | 399.1             | Batch deletes are more efficient for large operations |

#### ParallelSearch Performance

| Graph Size | Dimensions | Sequential (ns/op) | Parallel (ns/op) | Speedup |
|------------|-----------|-------------------|-----------------|---------|
| 100        | 128       | 19,617            | 17,967          | 1.09x   |
| 100        | 1536      | 122,608           | 111,925         | 1.10x   |
| 1000       | 128       | 28,467            | 28,267          | 1.01x   |
| 1000       | 1536      | 151,142           | 167,167         | 0.90x   |
| 10000      | 128       | 50,917            | 32,700          | 1.56x   |
| 10000      | 1536      | 273,525           | 287,167         | 0.95x   |
| 50000      | 1536      | 389,075           | 449,775         | 0.87x   |

These results show that:

1. **Concurrent Search**: Provides significant speedup for medium-sized graphs with moderate dimensions.
2. **Batch Operations**: Offer slight performance improvements and reduce lock contention.
3. **ParallelSearch**: Most effective for medium to large graphs with moderate dimensions.

### Thread Safety Design

The thread-safe implementation uses a read-write mutex pattern to allow multiple concurrent reads (searches) but exclusive writes (adds/deletes):

1. **Read-Write Lock**: The `Graph` struct contains a `sync.RWMutex` to protect shared data structures.

   ```go
   type Graph[K cmp.Ordered] struct {
       // ... other fields ...
       mu sync.RWMutex
       // ... other fields ...
   }
   ```

2. **Lock Usage**:
   * Read operations (`Search`, `Lookup`, `Dims`) use read locks (`RLock`/`RUnlock`)
   * Write operations (`Add`, `Delete`) use write locks (`Lock`/`Unlock`)
   * Batch operations (`BatchAdd`, `BatchSearch`, `BatchDelete`) use a single lock acquisition for multiple operations

3. **Deadlock Prevention**: Methods called from within locked methods (like `Len` and `Dims` when called from `Add`) avoid acquiring locks again to prevent deadlocks.

4. **Performance Considerations**:
   * Read locks allow multiple concurrent searches
   * Write locks ensure data consistency during modifications
   * Batch operations reduce lock contention for high-throughput scenarios

5. **Concurrency Patterns**:
   * For read-heavy workloads, the implementation performs well with minimal overhead
   * For write-heavy workloads, consider using batch operations to reduce lock contention
   * For mixed workloads, the implementation provides a good balance of safety and performance

This design ensures that the HNSW graph can be safely used in concurrent environments while maintaining reasonable performance characteristics.

### Batch Operations Design

Batch operations are designed to optimize performance for high-throughput scenarios by reducing lock contention:

1. **BatchAdd**: Adds multiple nodes in a single operation with a single lock acquisition.

   ```go
   func (g *Graph[K]) BatchAdd(nodes []Node[K]) error {
       g.mu.Lock()
       defer g.mu.Unlock()
       
       // Process all nodes in a batch...
   }
   ```

2. **BatchSearch**: Performs multiple searches in a single operation with a single lock acquisition.

   ```go
   func (g *Graph[K]) BatchSearch(queries []Vector, k int) ([][]Node[K], error) {
       g.mu.RLock()
       defer g.mu.RUnlock()
       
       // Process all queries in a batch...
   }
   ```

3. **BatchDelete**: Removes multiple nodes in a single operation with a single lock acquisition, returning a slice of booleans indicating success for each key.

   ```go
   func (g *Graph[K]) BatchDelete(keys []K) []bool {
       g.mu.Lock()
       defer g.mu.Unlock()
       
       // Process all deletions in a batch...
       // Return a slice of booleans indicating which keys were successfully deleted
   }
   ```

4. **Performance Benefits**:
   * **Reduced Lock Contention**: Acquiring the lock once for multiple operations reduces contention.
   * **Amortized Overhead**: The overhead of lock acquisition is amortized over multiple operations.
   * **Improved Throughput**: For high-throughput scenarios, batch operations can significantly improve performance.

5. **Use Cases**:
   * **Bulk Loading**: When loading a large number of vectors into the graph.
   * **Batch Processing**: When processing a batch of queries from a queue.
   * **High-Throughput APIs**: When serving a high volume of requests.
   * **Bulk Deletion**: When removing multiple vectors from the graph at once.

6. **Limitations**:
   * **Memory Usage**: Batch operations may require more memory to store intermediate results.
   * **Latency**: Individual operations within a batch may experience higher latency due to processing order.
   * **Error Handling**: An error in one operation may affect the entire batch.

Batch operations are particularly useful in scenarios where you need to process a large number of operations efficiently, such as bulk loading or high-throughput APIs.
