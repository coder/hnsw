// Package example provides examples of using the metadata extension for HNSW.
package example

import (
	"fmt"
	"log"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/meta"
)

// ProductMetadata represents metadata for a product.
type ProductMetadata struct {
	Name        string    `json:"name"`
	Category    string    `json:"category"`
	Price       float64   `json:"price"`
	Tags        []string  `json:"tags"`
	InStock     bool      `json:"inStock"`
	ReleaseDate time.Time `json:"releaseDate"`
}

// RunMetadataExample demonstrates the use of the metadata extension.
func RunMetadataExample() {
	// Create a graph and metadata store
	graph := hnsw.NewGraph[int]()
	store := meta.NewMemoryMetadataStore[int]()
	metadataGraph := meta.NewMetadataGraph(graph, store)

	// Create some product vectors and metadata
	products := []struct {
		ID       int
		Vector   []float32
		Metadata ProductMetadata
	}{
		{
			ID:     1,
			Vector: []float32{1.0, 0.0, 0.0},
			Metadata: ProductMetadata{
				Name:        "Smartphone X",
				Category:    "Electronics",
				Price:       999.99,
				Tags:        []string{"smartphone", "5G", "camera"},
				InStock:     true,
				ReleaseDate: time.Date(2023, 1, 15, 0, 0, 0, 0, time.UTC),
			},
		},
		{
			ID:     2,
			Vector: []float32{0.8, 0.2, 0.0},
			Metadata: ProductMetadata{
				Name:        "Tablet Pro",
				Category:    "Electronics",
				Price:       799.99,
				Tags:        []string{"tablet", "stylus", "portable"},
				InStock:     true,
				ReleaseDate: time.Date(2023, 3, 10, 0, 0, 0, 0, time.UTC),
			},
		},
		{
			ID:     3,
			Vector: []float32{0.0, 1.0, 0.0},
			Metadata: ProductMetadata{
				Name:        "Designer Jeans",
				Category:    "Clothing",
				Price:       129.99,
				Tags:        []string{"jeans", "denim", "fashion"},
				InStock:     true,
				ReleaseDate: time.Date(2023, 2, 5, 0, 0, 0, 0, time.UTC),
			},
		},
		{
			ID:     4,
			Vector: []float32{0.0, 0.8, 0.2},
			Metadata: ProductMetadata{
				Name:        "Running Shoes",
				Category:    "Footwear",
				Price:       89.99,
				Tags:        []string{"shoes", "running", "sports"},
				InStock:     false,
				ReleaseDate: time.Date(2022, 11, 20, 0, 0, 0, 0, time.UTC),
			},
		},
		{
			ID:     5,
			Vector: []float32{0.0, 0.0, 1.0},
			Metadata: ProductMetadata{
				Name:        "Coffee Maker",
				Category:    "Kitchen",
				Price:       149.99,
				Tags:        []string{"coffee", "kitchen", "appliance"},
				InStock:     true,
				ReleaseDate: time.Date(2023, 4, 5, 0, 0, 0, 0, time.UTC),
			},
		},
	}

	// Add products to the graph
	for _, product := range products {
		node := hnsw.MakeNode(product.ID, product.Vector)
		metadataNode, err := meta.NewMetadataNode(node, product.Metadata)
		if err != nil {
			log.Fatalf("Failed to create metadata node for product %d: %v", product.ID, err)
		}

		err = metadataGraph.Add(metadataNode)
		if err != nil {
			log.Fatalf("Failed to add product %d: %v", product.ID, err)
		}
	}

	fmt.Println("Products added to the graph with metadata")
	fmt.Println()

	// Example 1: Basic search
	fmt.Println("Example 1: Search for products similar to Smartphone X")
	query := []float32{1.0, 0.1, 0.0} // Similar to electronics
	results, err := metadataGraph.Search(query, 3)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		var metadata ProductMetadata
		err := result.GetMetadataAs(&metadata)
		if err != nil {
			log.Printf("Failed to get metadata for result %d: %v", i, err)
			continue
		}

		fmt.Printf("%d. %s - $%.2f (%s)\n", i+1, metadata.Name, metadata.Price, metadata.Category)
		fmt.Printf("   Tags: %v\n", metadata.Tags)
		fmt.Printf("   In Stock: %v, Released: %s\n", metadata.InStock, metadata.ReleaseDate.Format("2006-01-02"))
		fmt.Printf("   Distance: %.4f\n", result.Dist)
		fmt.Println()
	}

	// Example 2: Search with negative example
	fmt.Println("Example 2: Search for products similar to electronics but not like kitchen appliances")
	negative := []float32{0.0, 0.0, 1.0} // Coffee maker vector
	resultsWithNegative, err := metadataGraph.SearchWithNegative(query, negative, 3, 0.7)
	if err != nil {
		log.Fatalf("Search with negative failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range resultsWithNegative {
		var metadata ProductMetadata
		err := result.GetMetadataAs(&metadata)
		if err != nil {
			log.Printf("Failed to get metadata for result %d: %v", i, err)
			continue
		}

		fmt.Printf("%d. %s - $%.2f (%s)\n", i+1, metadata.Name, metadata.Price, metadata.Category)
		fmt.Printf("   Tags: %v\n", metadata.Tags)
		fmt.Printf("   In Stock: %v, Released: %s\n", metadata.InStock, metadata.ReleaseDate.Format("2006-01-02"))
		fmt.Printf("   Distance: %.4f\n", result.Dist)
		fmt.Println()
	}

	// Example 3: Batch search
	fmt.Println("Example 3: Batch search for multiple queries")
	queries := []hnsw.Vector{
		{1.0, 0.0, 0.0}, // Electronics
		{0.0, 1.0, 0.0}, // Clothing
	}
	batchResults, err := metadataGraph.BatchSearch(queries, 2)
	if err != nil {
		log.Fatalf("Batch search failed: %v", err)
	}

	for i, results := range batchResults {
		fmt.Printf("Results for query %d:\n", i+1)
		for j, result := range results {
			var metadata ProductMetadata
			err := result.GetMetadataAs(&metadata)
			if err != nil {
				log.Printf("Failed to get metadata for result %d: %v", j, err)
				continue
			}

			fmt.Printf("%d. %s - $%.2f (%s)\n", j+1, metadata.Name, metadata.Price, metadata.Category)
		}
		fmt.Println()
	}

	// Example 4: Get a specific product
	fmt.Println("Example 4: Get a specific product")
	productNode, ok := metadataGraph.Get(3)
	if !ok {
		log.Fatalf("Failed to get product 3")
	}

	var productMetadata ProductMetadata
	err = productNode.GetMetadataAs(&productMetadata)
	if err != nil {
		log.Fatalf("Failed to get metadata for product 3: %v", err)
	}

	fmt.Printf("Product: %s\n", productMetadata.Name)
	fmt.Printf("Category: %s\n", productMetadata.Category)
	fmt.Printf("Price: $%.2f\n", productMetadata.Price)
	fmt.Printf("Tags: %v\n", productMetadata.Tags)
	fmt.Printf("In Stock: %v\n", productMetadata.InStock)
	fmt.Printf("Release Date: %s\n", productMetadata.ReleaseDate.Format("2006-01-02"))
}
