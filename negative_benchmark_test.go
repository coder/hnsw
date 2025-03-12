package hnsw

import (
	"fmt"
	"math/rand"
	"testing"
)

// generateRandomVectors creates n random vectors of the specified dimension
func generateRandomVectors(n, dim int) []Vector {
	vectors := make([]Vector, n)
	for i := range vectors {
		vectors[i] = make(Vector, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()*2 - 1 // Random value between -1 and 1
		}
	}
	return vectors
}

// BenchmarkSearchWithNegative measures the performance of search with a negative example
func BenchmarkSearchWithNegative(b *testing.B) {
	dims := 128
	numNodes := 1000

	// Create a graph with random vectors
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
	vectors := generateRandomVectors(numNodes, dims)

	for i, vec := range vectors {
		g.Add(MakeNode(i, vec))
	}

	// Create query and negative vectors
	query := generateRandomVectors(1, dims)[0]
	negative := generateRandomVectors(1, dims)[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.SearchWithNegative(query, negative, 10, 0.5)
	}
}

// BenchmarkSearchWithNegatives measures the performance of search with multiple negative examples
func BenchmarkSearchWithNegatives(b *testing.B) {
	dims := 128
	numNodes := 1000
	numNegatives := 3

	// Create a graph with random vectors
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
	vectors := generateRandomVectors(numNodes, dims)

	for i, vec := range vectors {
		g.Add(MakeNode(i, vec))
	}

	// Create query and negative vectors
	query := generateRandomVectors(1, dims)[0]
	negatives := generateRandomVectors(numNegatives, dims)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.SearchWithNegatives(query, negatives, 10, 0.5)
	}
}

// BenchmarkBatchSearchWithNegatives measures the performance of batch search with negative examples
func BenchmarkBatchSearchWithNegatives(b *testing.B) {
	dims := 128
	numNodes := 1000
	batchSize := 10
	numNegativesPerQuery := 2

	// Create a graph with random vectors
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
	vectors := generateRandomVectors(numNodes, dims)

	for i, vec := range vectors {
		g.Add(MakeNode(i, vec))
	}

	// Create query and negative vectors
	queries := generateRandomVectors(batchSize, dims)

	// Create a set of negatives for each query
	negatives := make([][]Vector, batchSize)
	for i := range negatives {
		negatives[i] = generateRandomVectors(numNegativesPerQuery, dims)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.BatchSearchWithNegatives(queries, negatives, 10, 0.5)
	}
}

// BenchmarkCompareSearchMethods compares the performance of different search methods
func BenchmarkCompareSearchMethods(b *testing.B) {
	dims := 128
	numNodes := 1000

	// Create a graph with random vectors
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
	vectors := generateRandomVectors(numNodes, dims)

	for i, vec := range vectors {
		g.Add(MakeNode(i, vec))
	}

	// Create query and negative vectors
	query := generateRandomVectors(1, dims)[0]
	negative := generateRandomVectors(1, dims)[0]
	negatives := generateRandomVectors(3, dims)

	b.Run("Regular Search", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.Search(query, 10)
		}
	})

	b.Run("Search with Single Negative", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.SearchWithNegative(query, negative, 10, 0.5)
		}
	})

	b.Run("Search with Multiple Negatives", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.SearchWithNegatives(query, negatives, 10, 0.5)
		}
	})
}

// BenchmarkNegativeWeightImpact measures the impact of different negative weights
func BenchmarkNegativeWeightImpact(b *testing.B) {
	dims := 128
	numNodes := 1000

	// Create a graph with random vectors
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
	vectors := generateRandomVectors(numNodes, dims)

	for i, vec := range vectors {
		g.Add(MakeNode(i, vec))
	}

	// Create query and negative vectors
	query := generateRandomVectors(1, dims)[0]
	negative := generateRandomVectors(1, dims)[0]

	weights := []float32{0.1, 0.3, 0.5, 0.7, 0.9}

	for _, weight := range weights {
		b.Run(fmt.Sprintf("NegWeight=%.1f", weight), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g.SearchWithNegative(query, negative, 10, weight)
			}
		})
	}
}

// BenchmarkDimensionImpact measures the impact of vector dimensionality
func BenchmarkDimensionImpact(b *testing.B) {
	dimensions := []int{32, 128, 512, 1536}
	numNodes := 1000

	for _, dims := range dimensions {
		b.Run(fmt.Sprintf("Dims=%d", dims), func(b *testing.B) {
			// Create a graph with random vectors
			g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
			vectors := generateRandomVectors(numNodes, dims)

			for i, vec := range vectors {
				g.Add(MakeNode(i, vec))
			}

			// Create query and negative vectors
			query := generateRandomVectors(1, dims)[0]
			negative := generateRandomVectors(1, dims)[0]

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g.SearchWithNegative(query, negative, 10, 0.5)
			}
		})
	}
}
