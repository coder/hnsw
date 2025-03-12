package main

import (
	"fmt"
	"log"

	"github.com/TFMV/hnsw"
)

func main() {
	// Create a new graph with default configuration
	g, err := hnsw.NewGraphWithConfig[string](16, 0.25, 20, hnsw.CosineDistance)
	if err != nil {
		log.Fatalf("failed to create graph: %v", err)
	}

	// Add some vectors representing different concepts
	g.Add(
		hnsw.MakeNode("dog", []float32{1.0, 0.2, 0.1, 0.0}),
		hnsw.MakeNode("puppy", []float32{0.9, 0.3, 0.2, 0.1}),
		hnsw.MakeNode("canine", []float32{0.8, 0.3, 0.3, 0.0}),

		hnsw.MakeNode("cat", []float32{0.1, 1.0, 0.2, 0.0}),
		hnsw.MakeNode("kitten", []float32{0.2, 0.9, 0.3, 0.1}),
		hnsw.MakeNode("feline", []float32{0.3, 0.8, 0.3, 0.0}),

		hnsw.MakeNode("bird", []float32{0.1, 0.2, 1.0, 0.0}),
		hnsw.MakeNode("sparrow", []float32{0.2, 0.3, 0.9, 0.1}),
		hnsw.MakeNode("avian", []float32{0.3, 0.3, 0.8, 0.0}),

		hnsw.MakeNode("fish", []float32{0.0, 0.0, 0.0, 1.0}),
		hnsw.MakeNode("salmon", []float32{0.1, 0.1, 0.1, 0.9}),
		hnsw.MakeNode("aquatic", []float32{0.2, 0.2, 0.2, 0.8}),
	)

	// Example 1: Regular search for dog-related concepts
	fmt.Println("Example 1: Regular search for dog-related concepts")
	dogQuery := []float32{1.0, 0.2, 0.1, 0.0} // dog query
	results, err := g.Search(dogQuery, 3)
	if err != nil {
		log.Fatalf("failed to search: %v", err)
	}

	fmt.Println("Regular search results:")
	for i, result := range results {
		fmt.Printf("  %d. %s\n", i+1, result.Key)
	}
	fmt.Println()

	// Example 2: Search for dog-related concepts but not puppies
	fmt.Println("Example 2: Search for dog-related concepts but not puppies")
	puppyNegative := []float32{0.9, 0.3, 0.2, 0.1} // puppy (negative example)

	resultsWithNegative, err := g.SearchWithNegative(dogQuery, puppyNegative, 3, 0.5)
	if err != nil {
		log.Fatalf("failed to search with negative: %v", err)
	}

	fmt.Println("Search with negative example results:")
	for i, result := range resultsWithNegative {
		fmt.Printf("  %d. %s\n", i+1, result.Key)
	}
	fmt.Println()

	// Example 3: Search for pets but not dogs or cats
	fmt.Println("Example 3: Search for pets but not dogs or cats")
	petQuery := []float32{0.3, 0.3, 0.3, 0.3} // general pet query

	dogNegative := []float32{1.0, 0.2, 0.1, 0.0} // dog (negative example)
	catNegative := []float32{0.1, 1.0, 0.2, 0.0} // cat (negative example)

	negatives := []hnsw.Vector{dogNegative, catNegative}

	resultsWithMultipleNegatives, err := g.SearchWithNegatives(petQuery, negatives, 3, 0.7)
	if err != nil {
		log.Fatalf("failed to search with negatives: %v", err)
	}

	fmt.Println("Search with multiple negative examples results:")
	for i, result := range resultsWithMultipleNegatives {
		fmt.Printf("  %d. %s\n", i+1, result.Key)
	}
	fmt.Println()

	// Example 4: Batch search with negative examples
	fmt.Println("Example 4: Batch search with negative examples")
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

	fmt.Println("Batch search results:")
	for i, results := range batchResults {
		fmt.Printf("Query %d results:\n", i+1)
		for j, result := range results {
			fmt.Printf("  %d. %s\n", j+1, result.Key)
		}
	}
	fmt.Println()

	// Example 5: Impact of negative weight
	fmt.Println("Example 5: Impact of negative weight")

	// With low negative weight (0.1)
	lowWeightResults, err := g.SearchWithNegative(dogQuery, puppyNegative, 3, 0.1)
	if err != nil {
		log.Fatalf("failed to search with low negative weight: %v", err)
	}

	fmt.Println("Results with low negative weight (0.1):")
	for i, result := range lowWeightResults {
		fmt.Printf("  %d. %s\n", i+1, result.Key)
	}

	// With high negative weight (0.9)
	highWeightResults, err := g.SearchWithNegative(dogQuery, puppyNegative, 3, 0.9)
	if err != nil {
		log.Fatalf("failed to search with high negative weight: %v", err)
	}

	fmt.Println("Results with high negative weight (0.9):")
	for i, result := range highWeightResults {
		fmt.Printf("  %d. %s\n", i+1, result.Key)
	}
}
