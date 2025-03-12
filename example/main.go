package main

import (
	"fmt"
	"log"

	"github.com/TFMV/hnsw"
)

func main() {
	g, err := hnsw.NewGraphWithConfig[int](16, 0.25, 20, hnsw.EuclideanDistance)
	if err != nil {
		log.Fatalf("failed to create graph: %v", err)
	}
	g.Add(
		hnsw.MakeNode(1, []float32{1, 1, 1}),
		hnsw.MakeNode(2, []float32{1, -1, 0.999}),
		hnsw.MakeNode(3, []float32{1, 0, -0.5}),
	)

	neighbors, err := g.Search(
		[]float32{0.5, 0.5, 0.5},
		1,
	)
	if err != nil {
		log.Fatalf("failed to search graph: %v", err)
	}
	fmt.Printf("best friend: %v\n", neighbors[0].Value)
}
