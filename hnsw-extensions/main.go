// Package main provides examples of using the HNSW extensions.
package main

import (
	"fmt"
	"log"
	"os"

	docexample "github.com/TFMV/hnsw/hnsw-extensions/facets/examples/document_search"
	prodexample "github.com/TFMV/hnsw/hnsw-extensions/facets/examples/product_search"
	metaexample "github.com/TFMV/hnsw/hnsw-extensions/meta/example"
)

func main() {
	fmt.Println("HNSW Extensions Examples")
	fmt.Println("=======================")
	fmt.Println()

	// Get the current working directory
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get current working directory: %v", err)
	}
	fmt.Printf("Running examples from: %s\n\n", cwd)

	// Run the metadata example
	fmt.Println("Metadata Extension Example")
	fmt.Println("-----------------------")
	metaexample.RunMetadataExample()
	fmt.Println()

	// Run the facets document search example
	fmt.Println("Facets Document Search Example")
	fmt.Println("-----------------------------")
	docexample.DocumentSearch()
	fmt.Println()

	// Run the facets product search example
	fmt.Println("Facets Product Search Example")
	fmt.Println("----------------------------")
	prodexample.ProductSearch()
	fmt.Println()

	fmt.Println("All examples completed successfully!")
}
