package hnsw

// Analyzer is a struct that holds a graph and provides
// methods for analyzing it.
type Analyzer[T Embeddable] struct {
	Graph *Graph[T]
}

func (a *Analyzer[T]) Height() int {
	return len(a.Graph.layers)
}

// Connectivity returns the average number of edges in the
// graph for each non-empty layer.
func (a *Analyzer[T]) Connectivity() []float64 {
	var layerConnectivity []float64
	for _, layer := range a.Graph.layers {
		if len(layer.Nodes) == 0 {
			continue
		}

		var sum float64
		for _, node := range layer.Nodes {
			sum += float64(len(node.Neighbors))
		}

		layerConnectivity = append(layerConnectivity, sum/float64(len(layer.Nodes)))
	}

	return layerConnectivity
}

// Topography returns the number of nodes in each layer of the graph.
func (a *Analyzer[T]) Topography() []int {
	var topography []int
	for _, layer := range a.Graph.layers {
		topography = append(topography, len(layer.Nodes))
	}
	return topography
}
