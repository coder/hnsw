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
	)

	// Print initial graph size
	fmt.Printf("Initial graph size: %d\n", g.Len())

	// Example 1: Delete individual node
	fmt.Println("\nExample 1: Delete individual node")
	deleted := g.Delete("puppy")
	fmt.Printf("Deleted 'puppy': %v\n", deleted)
	fmt.Printf("Graph size after deleting 'puppy': %d\n", g.Len())

	// Example 2: Batch delete existing nodes
	fmt.Println("\nExample 2: Batch delete existing nodes")
	keysToDelete := []string{"dog", "cat", "bird"}
	results := g.BatchDelete(keysToDelete)

	fmt.Println("Batch delete results:")
	for i, key := range keysToDelete {
		fmt.Printf("  %s: %v\n", key, results[i])
	}
	fmt.Printf("Graph size after batch delete: %d\n", g.Len())

	// Example 3: Batch delete with mixed existing and non-existent nodes
	fmt.Println("\nExample 3: Batch delete with mixed existing and non-existent nodes")
	mixedKeys := []string{"canine", "unknown1", "kitten", "unknown2"}
	mixedResults := g.BatchDelete(mixedKeys)

	fmt.Println("Mixed batch delete results:")
	for i, key := range mixedKeys {
		fmt.Printf("  %s: %v\n", key, mixedResults[i])
	}
	fmt.Printf("Graph size after mixed batch delete: %d\n", g.Len())

	// Example 4: Search after deletions
	fmt.Println("\nExample 4: Search after deletions")
	query := []float32{0.3, 0.3, 0.8, 0.0} // Similar to "avian"
	searchResults, err := g.Search(query, 3)
	if err != nil {
		log.Fatalf("failed to search: %v", err)
	}

	fmt.Println("Search results after deletions:")
	for i, result := range searchResults {
		fmt.Printf("  %d. %s\n", i+1, result.Key)
	}

	// Example 5: Delete all remaining nodes
	fmt.Println("\nExample 5: Delete all remaining nodes")
	remainingKeys := []string{"feline", "sparrow", "avian"}
	finalResults := g.BatchDelete(remainingKeys)

	fmt.Println("Final batch delete results:")
	for i, key := range remainingKeys {
		fmt.Printf("  %s: %v\n", key, finalResults[i])
	}
	fmt.Printf("Final graph size: %d\n", g.Len())
}
