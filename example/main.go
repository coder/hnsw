package main

import (
	"fmt"
	"log"
	"sync"

	"github.com/TFMV/hnsw"
)

func main() {
	// Create a new graph with default configuration
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

	// Perform a basic search
	neighbors, err := g.Search(
		[]float32{0.5, 0.5, 0.5},
		1,
	)
	if err != nil {
		log.Fatalf("failed to search graph: %v", err)
	}
	fmt.Printf("best friend: %v\n", neighbors[0].Value)

	// Demonstrate concurrent operations
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

	// Concurrent adds
	wg.Add(numOperations)
	for i := 0; i < numOperations; i++ {
		go func(i int) {
			defer wg.Done()
			nodeID := 10 + i
			vector := []float32{float32(i), float32(i), float32(i)}
			err := g.Add(hnsw.MakeNode(nodeID, vector))
			if err != nil {
				log.Printf("Add error: %v", err)
				return
			}
		}(i)
	}

	// Wait for all operations to complete
	wg.Wait()

	// Verify the graph size after concurrent operations
	fmt.Printf("Graph size after concurrent operations: %d\n", g.Len())

	// Demonstrate batch operations
	batch := make([]hnsw.Node[int], 5)
	for i := range batch {
		nodeID := 100 + i
		vector := []float32{float32(i) * 0.5, float32(i) * 0.5, float32(i) * 0.5}
		batch[i] = hnsw.MakeNode(nodeID, vector)
	}

	// Add batch of nodes
	err = g.BatchAdd(batch)
	if err != nil {
		log.Fatalf("failed to batch add: %v", err)
	}

	// Batch search
	queries := [][]float32{
		{0.1, 0.1, 0.1},
		{0.2, 0.2, 0.2},
		{0.3, 0.3, 0.3},
	}
	batchResults, err := g.BatchSearch(queries, 2)
	if err != nil {
		log.Fatalf("failed to batch search: %v", err)
	}

	// Print batch search results
	for i, results := range batchResults {
		fmt.Printf("Batch search %d results: ", i)
		for _, node := range results {
			fmt.Printf("%d ", node.Key)
		}
		fmt.Println()
	}
}
