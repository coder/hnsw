package hnsw

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBatchDelete(t *testing.T) {
	// Create a test graph
	g, err := NewGraphWithConfig[int](16, 0.25, 20, CosineDistance)
	require.NoError(t, err)

	// Add test vectors
	for i := 1; i <= 10; i++ {
		g.Add(MakeNode(i, []float32{float32(i), float32(i), float32(i)}))
	}

	// Verify initial graph size
	assert.Equal(t, 10, g.Len(), "Graph should have 10 nodes initially")

	t.Run("Delete existing nodes", func(t *testing.T) {
		// Delete nodes 1, 3, 5
		keysToDelete := []int{1, 3, 5}
		results := g.BatchDelete(keysToDelete)

		// Verify results
		assert.Equal(t, []bool{true, true, true}, results, "All deletions should be successful")

		// Verify graph size
		assert.Equal(t, 7, g.Len(), "Graph should have 7 nodes after deletion")

		// Verify nodes are deleted
		for _, key := range keysToDelete {
			_, exists := g.Lookup(key)
			assert.False(t, exists, "Node %d should be deleted", key)
		}

		// Verify other nodes still exist
		for _, key := range []int{2, 4, 6, 7, 8, 9, 10} {
			_, exists := g.Lookup(key)
			assert.True(t, exists, "Node %d should still exist", key)
		}
	})

	t.Run("Delete non-existent nodes", func(t *testing.T) {
		// Try to delete nodes that don't exist
		keysToDelete := []int{11, 12, 13}
		results := g.BatchDelete(keysToDelete)

		// Verify results
		assert.Equal(t, []bool{false, false, false}, results, "All deletions should fail")

		// Verify graph size remains unchanged
		assert.Equal(t, 7, g.Len(), "Graph size should remain unchanged")
	})

	t.Run("Delete mixed existing and non-existent nodes", func(t *testing.T) {
		// Delete a mix of existing and non-existent nodes
		keysToDelete := []int{2, 15, 4, 20}
		results := g.BatchDelete(keysToDelete)

		// Verify results
		assert.Equal(t, []bool{true, false, true, false}, results, "Only existing nodes should be deleted")

		// Verify graph size
		assert.Equal(t, 5, g.Len(), "Graph should have 5 nodes after deletion")

		// Verify nodes are deleted
		for _, key := range []int{2, 4} {
			_, exists := g.Lookup(key)
			assert.False(t, exists, "Node %d should be deleted", key)
		}

		// Verify other nodes still exist
		for _, key := range []int{6, 7, 8, 9, 10} {
			_, exists := g.Lookup(key)
			assert.True(t, exists, "Node %d should still exist", key)
		}
	})

	t.Run("Delete with empty slice", func(t *testing.T) {
		// Delete with empty slice
		results := g.BatchDelete([]int{})

		// Verify results
		assert.Equal(t, []bool{}, results, "Result should be empty slice")

		// Verify graph size remains unchanged
		assert.Equal(t, 5, g.Len(), "Graph size should remain unchanged")
	})

	t.Run("Delete all remaining nodes", func(t *testing.T) {
		// Delete all remaining nodes
		keysToDelete := []int{6, 7, 8, 9, 10}
		results := g.BatchDelete(keysToDelete)

		// Verify results
		assert.Equal(t, []bool{true, true, true, true, true}, results, "All deletions should be successful")

		// Verify graph is empty
		assert.Equal(t, 0, g.Len(), "Graph should be empty")
	})
}

func TestBatchDeleteConcurrency(t *testing.T) {
	// Create a test graph
	g, err := NewGraphWithConfig[int](16, 0.25, 20, CosineDistance)
	require.NoError(t, err)

	// Add test vectors
	for i := 1; i <= 100; i++ {
		g.Add(MakeNode(i, []float32{float32(i), float32(i), float32(i)}))
	}

	// Verify initial graph size
	assert.Equal(t, 100, g.Len(), "Graph should have 100 nodes initially")

	// Create batches of keys to delete
	batch1 := make([]int, 20)
	batch2 := make([]int, 20)
	batch3 := make([]int, 20)

	for i := 0; i < 20; i++ {
		batch1[i] = i + 1  // 1-20
		batch2[i] = i + 41 // 41-60
		batch3[i] = i + 81 // 81-100
	}

	// Delete batches concurrently
	results1 := g.BatchDelete(batch1)
	results2 := g.BatchDelete(batch2)
	results3 := g.BatchDelete(batch3)

	// Verify all deletions were successful
	for i, success := range results1 {
		assert.True(t, success, "Deletion of node %d should be successful", batch1[i])
	}
	for i, success := range results2 {
		assert.True(t, success, "Deletion of node %d should be successful", batch2[i])
	}
	for i, success := range results3 {
		assert.True(t, success, "Deletion of node %d should be successful", batch3[i])
	}

	// Verify graph size
	assert.Equal(t, 40, g.Len(), "Graph should have 40 nodes after batch deletions")

	// Verify deleted nodes are gone
	for i := 1; i <= 100; i++ {
		_, exists := g.Lookup(i)
		if (i >= 1 && i <= 20) || (i >= 41 && i <= 60) || (i >= 81 && i <= 100) {
			assert.False(t, exists, "Node %d should be deleted", i)
		} else {
			assert.True(t, exists, "Node %d should still exist", i)
		}
	}
}

func BenchmarkBatchDelete(b *testing.B) {
	// Create a graph with random vectors
	g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)

	// Add 1000 nodes
	for i := 0; i < 1000; i++ {
		vector := make(Vector, 128)
		for j := range vector {
			vector[j] = float32(i) * 0.01
		}
		g.Add(MakeNode(i, vector))
	}

	// Create batches of different sizes for benchmarking
	smallBatch := make([]int, 10)
	mediumBatch := make([]int, 100)
	largeBatch := make([]int, 500)

	for i := 0; i < 10; i++ {
		smallBatch[i] = i
	}
	for i := 0; i < 100; i++ {
		mediumBatch[i] = i + 200
	}
	for i := 0; i < 500; i++ {
		largeBatch[i] = i + 400
	}

	b.Run("Individual Deletes", func(b *testing.B) {
		// Reset the graph for each benchmark
		g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
		for i := 0; i < 1000; i++ {
			vector := make(Vector, 128)
			for j := range vector {
				vector[j] = float32(i) * 0.01
			}
			g.Add(MakeNode(i, vector))
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < 10; j++ {
				g.Delete(j)
			}
		}
	})

	b.Run("Small Batch (10)", func(b *testing.B) {
		// Reset the graph for each benchmark
		g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
		for i := 0; i < 1000; i++ {
			vector := make(Vector, 128)
			for j := range vector {
				vector[j] = float32(i) * 0.01
			}
			g.Add(MakeNode(i, vector))
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.BatchDelete(smallBatch)
		}
	})

	b.Run("Medium Batch (100)", func(b *testing.B) {
		// Reset the graph for each benchmark
		g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
		for i := 0; i < 1000; i++ {
			vector := make(Vector, 128)
			for j := range vector {
				vector[j] = float32(i) * 0.01
			}
			g.Add(MakeNode(i, vector))
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.BatchDelete(mediumBatch)
		}
	})

	b.Run("Large Batch (500)", func(b *testing.B) {
		// Reset the graph for each benchmark
		g, _ := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
		for i := 0; i < 1000; i++ {
			vector := make(Vector, 128)
			for j := range vector {
				vector[j] = float32(i) * 0.01
			}
			g.Add(MakeNode(i, vector))
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.BatchDelete(largeBatch)
		}
	})
}
