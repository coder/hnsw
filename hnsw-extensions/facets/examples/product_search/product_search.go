// Package examples provides example implementations of the hnsw-extensions.
package examples

import (
	"fmt"
	"log"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
)

// Product represents a product with various attributes.
type Product struct {
	ID          int
	Name        string
	Description string
	Price       float64
	Category    string
	Brand       string
	Tags        []string
	Vector      []float32 // Embedding vector
}

// ProductSearch demonstrates how to use faceted search for product search.
func ProductSearch() {
	// Create a new HNSW graph
	graph := hnsw.NewGraph[int]()

	// Create a facet store
	store := facets.NewMemoryFacetStore[int]()

	// Create a faceted graph
	facetedGraph := facets.NewFacetedGraph(graph, store)

	// Add some products
	products := []Product{
		{
			ID:          1,
			Name:        "Smartphone X",
			Description: "Latest smartphone with advanced features",
			Price:       999.99,
			Category:    "Electronics",
			Brand:       "TechCo",
			Tags:        []string{"smartphone", "tech", "mobile"},
			Vector:      []float32{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		{
			ID:          2,
			Name:        "Laptop Pro",
			Description: "High-performance laptop for professionals",
			Price:       1499.99,
			Category:    "Electronics",
			Brand:       "TechCo",
			Tags:        []string{"laptop", "tech", "computer"},
			Vector:      []float32{0.2, 0.3, 0.4, 0.5, 0.6},
		},
		{
			ID:          3,
			Name:        "Wireless Headphones",
			Description: "Noise-cancelling wireless headphones",
			Price:       299.99,
			Category:    "Electronics",
			Brand:       "AudioTech",
			Tags:        []string{"headphones", "audio", "wireless"},
			Vector:      []float32{0.3, 0.4, 0.5, 0.6, 0.7},
		},
		{
			ID:          4,
			Name:        "Running Shoes",
			Description: "Comfortable shoes for running",
			Price:       129.99,
			Category:    "Footwear",
			Brand:       "SportyBrand",
			Tags:        []string{"shoes", "running", "sports"},
			Vector:      []float32{0.5, 0.6, 0.7, 0.8, 0.9},
		},
		{
			ID:          5,
			Name:        "Fitness Tracker",
			Description: "Track your fitness activities",
			Price:       199.99,
			Category:    "Electronics",
			Brand:       "SportyTech",
			Tags:        []string{"fitness", "wearable", "tech"},
			Vector:      []float32{0.4, 0.5, 0.6, 0.7, 0.8},
		},
	}

	// Add products to the faceted graph
	for _, product := range products {
		// Create HNSW node
		node := hnsw.MakeNode(product.ID, product.Vector)

		// Create facets
		productFacets := []facets.Facet{
			facets.NewBasicFacet("name", product.Name),
			facets.NewBasicFacet("price", product.Price),
			facets.NewBasicFacet("category", product.Category),
			facets.NewBasicFacet("brand", product.Brand),
		}

		// Add tags as separate facets
		for _, tag := range product.Tags {
			productFacets = append(productFacets, facets.NewBasicFacet("tag", tag))
		}

		// Create faceted node
		facetedNode := facets.NewFacetedNode(node, productFacets)

		// Add to faceted graph
		err := facetedGraph.Add(facetedNode)
		if err != nil {
			log.Fatalf("Failed to add product %d: %v", product.ID, err)
		}
	}

	// Example 1: Search for products similar to "Fitness Tracker" with price filter
	fmt.Println("Example 1: Search for products similar to 'Fitness Tracker' with price filter")
	queryVector := products[4].Vector // Fitness Tracker vector
	priceFilter := facets.NewRangeFilter("price", 0, 300)

	results, err := facetedGraph.Search(
		queryVector,
		[]facets.FacetFilter{priceFilter},
		3,
		2,
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		productID := result.Node.Key
		product := findProductByID(products, productID)
		fmt.Printf("%d. %s - $%.2f (%s)\n", i+1, product.Name, product.Price, product.Category)
	}
	fmt.Println()

	// Example 2: Search for electronics with brand filter
	fmt.Println("Example 2: Search for electronics with brand filter")
	categoryFilter := facets.NewEqualityFilter("category", "Electronics")
	brandFilter := facets.NewEqualityFilter("brand", "TechCo")

	results, err = facetedGraph.Search(
		queryVector,
		[]facets.FacetFilter{categoryFilter, brandFilter},
		3,
		2,
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		productID := result.Node.Key
		product := findProductByID(products, productID)
		fmt.Printf("%d. %s - $%.2f (%s, %s)\n", i+1, product.Name, product.Price, product.Category, product.Brand)
	}
	fmt.Println()

	// Example 3: Search with negative example (avoid laptops)
	fmt.Println("Example 3: Search with negative example (avoid laptops)")
	negativeVector := products[1].Vector // Laptop Pro vector

	results, err = facetedGraph.SearchWithNegative(
		queryVector,
		negativeVector,
		[]facets.FacetFilter{categoryFilter},
		3,
		0.7,
		2,
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		productID := result.Node.Key
		product := findProductByID(products, productID)
		fmt.Printf("%d. %s - $%.2f (%s)\n", i+1, product.Name, product.Price, product.Category)
	}
	fmt.Println()

	// Example 4: Get facet aggregations
	fmt.Println("Example 4: Get facet aggregations")
	aggregations, err := facetedGraph.GetFacetAggregations(
		queryVector,
		[]facets.FacetFilter{},
		[]string{"category", "brand"},
		5,
		1,
	)
	if err != nil {
		log.Fatalf("Aggregation failed: %v", err)
	}

	fmt.Println("Category aggregations:")
	for value, count := range aggregations["category"].Values {
		fmt.Printf("- %s: %d products\n", value, count)
	}

	fmt.Println("Brand aggregations:")
	for value, count := range aggregations["brand"].Values {
		fmt.Printf("- %s: %d products\n", value, count)
	}
}

// findProductByID finds a product by its ID.
func findProductByID(products []Product, id int) Product {
	for _, product := range products {
		if product.ID == id {
			return product
		}
	}
	return Product{}
}
