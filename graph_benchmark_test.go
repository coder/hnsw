package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
)

// generateRandomVector creates a random vector of the specified dimension
func generateRandomVector(dim int) Vector {
	vec := make(Vector, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // Random value between -1 and 1
	}
	return vec
}

// BenchmarkSequentialAdd measures the performance of sequential Add operations
func BenchmarkSequentialAdd(b *testing.B) {
	dims := 128
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}
}

// BenchmarkConcurrentAdd measures the performance of concurrent Add operations
func BenchmarkConcurrentAdd(b *testing.B) {
	dims := 128
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Use atomic counter for thread-safe ID generation
	var counter int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Use atomic increment to get unique IDs
			id := int(atomic.AddInt64(&counter, 1))
			g.Add(MakeNode(id, generateRandomVector(dims)))
		}
	})
}

// BenchmarkSequentialSearch measures the performance of sequential Search operations
func BenchmarkSequentialSearch(b *testing.B) {
	dims := 128
	numNodes := 1000
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add nodes to the graph
	for i := 0; i < numNodes; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}

	// Create a query vector
	queryVec := generateRandomVector(dims)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Search(queryVec, 10)
	}
}

// BenchmarkConcurrentSearch measures the performance of concurrent Search operations
func BenchmarkConcurrentSearch(b *testing.B) {
	dims := 128
	numNodes := 1000
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add nodes to the graph
	for i := 0; i < numNodes; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}

	// Create a query vector
	queryVec := generateRandomVector(dims)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			g.Search(queryVec, 10)
		}
	})
}

// BenchmarkMixedOperations measures the performance of mixed Add and Search operations
func BenchmarkMixedOperations(b *testing.B) {
	dims := 128
	numNodes := 100
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add initial nodes to the graph
	for i := 0; i < numNodes; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}

	// Use atomic counter for thread-safe ID generation
	counter := int64(numNodes)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		localRand := rand.New(rand.NewSource(rand.Int63()))
		for pb.Next() {
			if localRand.Float32() < 0.2 { // 20% adds, 80% searches
				// Add operation
				id := int(atomic.AddInt64(&counter, 1))
				g.Add(MakeNode(id, generateRandomVector(dims)))
			} else {
				// Search operation
				queryVec := generateRandomVector(dims)
				g.Search(queryVec, 10)
			}
		}
	})
}

// TestConcurrentSafety verifies that the implementation is thread-safe
func TestConcurrentSafety(t *testing.T) {
	dims := 128
	numNodes := 1000
	numOperations := 10000
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add initial nodes to the graph
	for i := 0; i < numNodes; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}

	var wg sync.WaitGroup
	wg.Add(numOperations)

	// Run concurrent operations
	for i := 0; i < numOperations; i++ {
		go func(id int) {
			defer wg.Done()
			if id%5 == 0 { // 20% adds
				g.Add(MakeNode(numNodes+id, generateRandomVector(dims)))
			} else if id%20 == 1 { // 5% deletes
				g.Delete(id % numNodes)
			} else { // 75% searches
				queryVec := generateRandomVector(dims)
				g.Search(queryVec, 10)
			}
		}(i)
	}

	wg.Wait()

	// Verify the graph is still valid
	if err := g.Validate(); err != nil {
		t.Errorf("Graph validation failed after concurrent operations: %v", err)
	}

	// Verify we can still perform operations
	_, err := g.Search(generateRandomVector(dims), 10)
	if err != nil {
		t.Errorf("Search failed after concurrent operations: %v", err)
	}

	fmt.Printf("Graph size after concurrent operations: %d\n", g.Len())
}

// BenchmarkBatchAdd measures the performance of batch Add operations
func BenchmarkBatchAdd(b *testing.B) {
	dims := 128
	batchSize := 100

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

		// Create a batch of nodes
		batch := make([]Node[int], batchSize)
		for j := 0; j < batchSize; j++ {
			batch[j] = MakeNode(j, generateRandomVector(dims))
		}

		// Add the batch
		g.BatchAdd(batch)
	}
}

// BenchmarkIndividualAdds measures the performance of individual Add operations
func BenchmarkIndividualAdds(b *testing.B) {
	dims := 128
	batchSize := 100

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

		// Add nodes individually
		for j := 0; j < batchSize; j++ {
			g.Add(MakeNode(j, generateRandomVector(dims)))
		}
	}
}

// BenchmarkBatchSearch measures the performance of batch Search operations
func BenchmarkBatchSearch(b *testing.B) {
	dims := 128
	numNodes := 1000
	batchSize := 100

	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add nodes to the graph
	for i := 0; i < numNodes; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}

	// Create query vectors
	queries := make([]Vector, batchSize)
	for i := range queries {
		queries[i] = generateRandomVector(dims)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.BatchSearch(queries, 10)
	}
}

// BenchmarkIndividualSearches measures the performance of individual Search operations
func BenchmarkIndividualSearches(b *testing.B) {
	dims := 128
	numNodes := 1000
	batchSize := 100

	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add nodes to the graph
	for i := 0; i < numNodes; i++ {
		g.Add(MakeNode(i, generateRandomVector(dims)))
	}

	// Create query vectors
	queries := make([]Vector, batchSize)
	for i := range queries {
		queries[i] = generateRandomVector(dims)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, query := range queries {
			g.Search(query, 10)
		}
	}
}
