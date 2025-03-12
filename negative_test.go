package hnsw

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSearchWithNegative(t *testing.T) {
	// Create a test graph
	g, err := NewGraphWithConfig[int](16, 0.25, 20, CosineDistance)
	require.NoError(t, err)

	// Add test vectors
	// Group 1: Vectors related to "dog"
	g.Add(MakeNode(1, []float32{1.0, 0.2, 0.1})) // dog
	g.Add(MakeNode(2, []float32{0.9, 0.3, 0.2})) // puppy
	g.Add(MakeNode(3, []float32{0.8, 0.3, 0.3})) // canine

	// Group 2: Vectors related to "cat"
	g.Add(MakeNode(4, []float32{0.1, 1.0, 0.2})) // cat
	g.Add(MakeNode(5, []float32{0.2, 0.9, 0.3})) // kitten
	g.Add(MakeNode(6, []float32{0.3, 0.8, 0.3})) // feline

	// Group 3: Vectors related to "bird"
	g.Add(MakeNode(7, []float32{0.1, 0.2, 1.0})) // bird
	g.Add(MakeNode(8, []float32{0.2, 0.3, 0.9})) // sparrow
	g.Add(MakeNode(9, []float32{0.3, 0.3, 0.8})) // avian

	t.Run("Basic search without negative", func(t *testing.T) {
		// Search for dog-related vectors
		query := []float32{1.0, 0.2, 0.1} // dog query
		results, err := g.Search(query, 3)
		require.NoError(t, err)
		require.Len(t, results, 3)

		// Results should be dog-related
		ids := []int{results[0].Key, results[1].Key, results[2].Key}
		assert.Contains(t, ids, 1) // dog
		assert.Contains(t, ids, 2) // puppy
		assert.Contains(t, ids, 3) // canine
	})

	t.Run("Search with negative example", func(t *testing.T) {
		// Search for dog-related vectors but not puppy
		query := []float32{1.0, 0.2, 0.1}    // dog query
		negative := []float32{0.9, 0.3, 0.2} // puppy (negative example)

		results, err := g.SearchWithNegative(query, negative, 3, 0.5)
		require.NoError(t, err)
		require.Len(t, results, 3)

		// Results should include dog but not puppy
		assert.Equal(t, 1, results[0].Key)    // dog should be first
		assert.NotEqual(t, 2, results[0].Key) // puppy should not be first

		// Check if puppy is ranked lower than without negative example
		regularResults, _ := g.Search(query, 9)

		// Find positions in regular search
		regularPuppyPos := -1
		for i, r := range regularResults {
			if r.Key == 2 { // puppy
				regularPuppyPos = i
				break
			}
		}

		// Find positions in negative search
		negativePuppyPos := -1
		for i, r := range results {
			if r.Key == 2 { // puppy
				negativePuppyPos = i
				break
			}
		}

		// If puppy is in both result sets, it should be ranked lower in the negative search
		if negativePuppyPos != -1 && regularPuppyPos != -1 {
			assert.Greater(t, negativePuppyPos, regularPuppyPos, "Puppy should be ranked lower with negative example")
		}
	})

	t.Run("Search with multiple negative examples", func(t *testing.T) {
		// Search for animal-related vectors but not dog or cat related
		query := []float32{0.4, 0.4, 0.4} // general animal query
		negatives := []Vector{
			{1.0, 0.2, 0.1}, // dog (negative example)
			{0.1, 1.0, 0.2}, // cat (negative example)
		}

		results, err := g.SearchWithNegatives(query, negatives, 3, 0.7)
		require.NoError(t, err)
		require.Len(t, results, 3)

		// Results should prioritize bird-related vectors
		birdRelated := 0
		for _, r := range results {
			if r.Key >= 7 && r.Key <= 9 {
				birdRelated++
			}
		}

		assert.GreaterOrEqual(t, birdRelated, 1, "At least one bird-related vector should be in results")
	})

	t.Run("Negative weight impact", func(t *testing.T) {
		query := []float32{1.0, 0.2, 0.1}    // dog query
		negative := []float32{0.9, 0.3, 0.2} // puppy (negative example)

		// With low negative weight
		lowResults, err := g.SearchWithNegative(query, negative, 3, 0.1)
		require.NoError(t, err)

		// With high negative weight
		highResults, err := g.SearchWithNegative(query, negative, 3, 0.9)
		require.NoError(t, err)

		// Check if puppy is ranked differently based on negative weight
		lowPuppyPos := -1
		for i, r := range lowResults {
			if r.Key == 2 { // puppy
				lowPuppyPos = i
				break
			}
		}

		highPuppyPos := -1
		for i, r := range highResults {
			if r.Key == 2 { // puppy
				highPuppyPos = i
				break
			}
		}

		// If puppy is in both result sets, it should be ranked lower with higher negative weight
		if lowPuppyPos != -1 && highPuppyPos != -1 {
			assert.Greater(t, highPuppyPos, lowPuppyPos, "Puppy should be ranked lower with higher negative weight")
		} else if lowPuppyPos != -1 {
			// Puppy might be excluded entirely with high negative weight
			assert.Equal(t, -1, highPuppyPos, "Puppy should be excluded with high negative weight")
		}
	})
}

func TestBatchSearchWithNegatives(t *testing.T) {
	// Create a test graph
	g, err := NewGraphWithConfig[int](16, 0.25, 20, CosineDistance)
	require.NoError(t, err)

	// Add test vectors
	// Group 1: Vectors related to "dog"
	g.Add(MakeNode(1, []float32{1.0, 0.2, 0.1})) // dog
	g.Add(MakeNode(2, []float32{0.9, 0.3, 0.2})) // puppy
	g.Add(MakeNode(3, []float32{0.8, 0.3, 0.3})) // canine

	// Group 2: Vectors related to "cat"
	g.Add(MakeNode(4, []float32{0.1, 1.0, 0.2})) // cat
	g.Add(MakeNode(5, []float32{0.2, 0.9, 0.3})) // kitten
	g.Add(MakeNode(6, []float32{0.3, 0.8, 0.3})) // feline

	// Group 3: Vectors related to "bird"
	g.Add(MakeNode(7, []float32{0.1, 0.2, 1.0})) // bird
	g.Add(MakeNode(8, []float32{0.2, 0.3, 0.9})) // sparrow
	g.Add(MakeNode(9, []float32{0.3, 0.3, 0.8})) // avian

	t.Run("Batch search with negatives", func(t *testing.T) {
		queries := []Vector{
			{1.0, 0.2, 0.1}, // dog query
			{0.1, 1.0, 0.2}, // cat query
		}

		negatives := [][]Vector{
			{
				{0.9, 0.3, 0.2}, // puppy (negative for dog query)
			},
			{
				{0.2, 0.9, 0.3}, // kitten (negative for cat query)
			},
		}

		results, err := g.BatchSearchWithNegatives(queries, negatives, 3, 0.5)
		require.NoError(t, err)
		require.Len(t, results, 2)

		// Check first query results (dog with puppy as negative)
		require.Len(t, results[0], 3)
		assert.Equal(t, 1, results[0][0].Key) // dog should be first

		// Check second query results (cat with kitten as negative)
		require.Len(t, results[1], 3)
		assert.Equal(t, 4, results[1][0].Key) // cat should be first
	})

	t.Run("Batch search with empty negatives", func(t *testing.T) {
		queries := []Vector{
			{1.0, 0.2, 0.1}, // dog query
			{0.1, 1.0, 0.2}, // cat query
		}

		negatives := [][]Vector{
			{}, // No negatives for dog query
			{
				{0.2, 0.9, 0.3}, // kitten (negative for cat query)
			},
		}

		results, err := g.BatchSearchWithNegatives(queries, negatives, 3, 0.5)
		require.NoError(t, err)
		require.Len(t, results, 2)

		// Check first query results (dog with no negatives)
		require.Len(t, results[0], 3)
		assert.Equal(t, 1, results[0][0].Key) // dog should be first

		// Check second query results (cat with kitten as negative)
		require.Len(t, results[1], 3)
		assert.Equal(t, 4, results[1][0].Key) // cat should be first
	})

	t.Run("Batch search with multiple negatives", func(t *testing.T) {
		queries := []Vector{
			{0.4, 0.4, 0.4}, // general animal query
		}

		negatives := [][]Vector{
			{
				{1.0, 0.2, 0.1}, // dog (negative example)
				{0.1, 1.0, 0.2}, // cat (negative example)
			},
		}

		results, err := g.BatchSearchWithNegatives(queries, negatives, 3, 0.7)
		require.NoError(t, err)
		require.Len(t, results, 1)
		require.Len(t, results[0], 3)

		// Results should prioritize bird-related vectors
		birdRelated := 0
		for _, r := range results[0] {
			if r.Key >= 7 && r.Key <= 9 {
				birdRelated++
			}
		}

		assert.GreaterOrEqual(t, birdRelated, 1, "At least one bird-related vector should be in results")
	})
}
