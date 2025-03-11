package hnsw

import (
	"cmp"
	"math"
)

// Analyzer is a struct that holds a graph and provides
// methods for analyzing it. It offers no compatibility guarantee
// as the methods of measuring the graph's health with change
// with the implementation.
type Analyzer[K cmp.Ordered] struct {
	Graph *Graph[K]
}

func (a *Analyzer[T]) Height() int {
	return len(a.Graph.layers)
}

// Connectivity returns the average number of edges in the
// graph for each non-empty layer.
func (a *Analyzer[T]) Connectivity() []float64 {
	var layerConnectivity []float64
	for _, layer := range a.Graph.layers {
		if len(layer.nodes) == 0 {
			continue
		}

		var sum float64
		for _, node := range layer.nodes {
			sum += float64(len(node.neighbors))
		}

		layerConnectivity = append(layerConnectivity, sum/float64(len(layer.nodes)))
	}

	return layerConnectivity
}

// Topography returns the number of nodes in each layer of the graph.
func (a *Analyzer[T]) Topography() []int {
	var topography []int
	for _, layer := range a.Graph.layers {
		topography = append(topography, len(layer.nodes))
	}
	return topography
}

// QualityMetrics calculates various quality metrics for the graph.
// Returns a struct containing metrics that evaluate the graph's quality.
func (a *Analyzer[T]) QualityMetrics() GraphQualityMetrics {
	if len(a.Graph.layers) == 0 {
		return GraphQualityMetrics{}
	}

	baseLayer := a.Graph.layers[0]
	metrics := GraphQualityMetrics{
		NodeCount:          len(baseLayer.nodes),
		AvgConnectivity:    a.averageConnectivity(),
		ConnectivityStdDev: a.connectivityStdDev(),
		DistortionRatio:    a.calculateDistortionRatio(),
		LayerBalance:       a.calculateLayerBalance(),
		GraphHeight:        len(a.Graph.layers),
	}

	return metrics
}

// GraphQualityMetrics contains various metrics that evaluate the quality of the graph.
type GraphQualityMetrics struct {
	// NodeCount is the total number of nodes in the graph.
	NodeCount int

	// AvgConnectivity is the average number of connections per node in the base layer.
	AvgConnectivity float64

	// ConnectivityStdDev is the standard deviation of connections per node.
	ConnectivityStdDev float64

	// DistortionRatio measures how well the graph preserves distances.
	// Lower values indicate better distance preservation.
	DistortionRatio float64

	// LayerBalance measures how well balanced the layers are.
	// Values closer to 1.0 indicate better balance.
	LayerBalance float64

	// GraphHeight is the number of layers in the graph.
	GraphHeight int
}

// averageConnectivity calculates the average number of connections per node in the base layer.
func (a *Analyzer[T]) averageConnectivity() float64 {
	if len(a.Graph.layers) == 0 {
		return 0
	}

	baseLayer := a.Graph.layers[0]
	if len(baseLayer.nodes) == 0 {
		return 0
	}

	var sum float64
	for _, node := range baseLayer.nodes {
		sum += float64(len(node.neighbors))
	}

	return sum / float64(len(baseLayer.nodes))
}

// connectivityStdDev calculates the standard deviation of connections per node.
func (a *Analyzer[T]) connectivityStdDev() float64 {
	if len(a.Graph.layers) == 0 {
		return 0
	}

	baseLayer := a.Graph.layers[0]
	if len(baseLayer.nodes) == 0 {
		return 0
	}

	avg := a.averageConnectivity()
	var sumSquaredDiff float64
	for _, node := range baseLayer.nodes {
		diff := float64(len(node.neighbors)) - avg
		sumSquaredDiff += diff * diff
	}

	return math.Sqrt(sumSquaredDiff / float64(len(baseLayer.nodes)))
}

// calculateDistortionRatio estimates how well the graph preserves distances.
// It samples a subset of nodes and compares graph distance to actual distance.
// Lower values indicate better distance preservation.
func (a *Analyzer[T]) calculateDistortionRatio() float64 {
	if len(a.Graph.layers) == 0 || a.Graph.layers[0].size() < 10 {
		return 0
	}

	baseLayer := a.Graph.layers[0]
	nodeCount := len(baseLayer.nodes)

	// Sample size - use at most 100 nodes to keep computation reasonable
	sampleSize := min(100, nodeCount)
	if sampleSize < 5 {
		return 0
	}

	// Sample nodes
	sampledNodes := make([]*layerNode[T], 0, sampleSize)
	i := 0
	for _, node := range baseLayer.nodes {
		if i >= sampleSize {
			break
		}
		sampledNodes = append(sampledNodes, node)
		i++
	}

	var distortionSum float64
	var pairsCount int

	// For each pair of sampled nodes
	for i := 0; i < len(sampledNodes); i++ {
		for j := i + 1; j < len(sampledNodes); j++ {
			nodeA := sampledNodes[i]
			nodeB := sampledNodes[j]

			// Calculate actual distance
			actualDist := a.Graph.Distance(nodeA.Value, nodeB.Value)

			// Estimate graph distance (number of hops)
			graphDist := a.estimateGraphDistance(nodeA, nodeB)

			if graphDist > 0 && !math.IsNaN(float64(actualDist)) && !math.IsInf(float64(actualDist), 0) {
				// Calculate distortion as ratio between graph distance and actual distance
				distortion := float64(graphDist) / float64(actualDist)
				distortionSum += distortion
				pairsCount++
			}
		}
	}

	if pairsCount == 0 {
		return 0
	}

	return distortionSum / float64(pairsCount)
}

// estimateGraphDistance estimates the number of hops between two nodes.
// Returns the number of hops or -1 if no path is found.
func (a *Analyzer[T]) estimateGraphDistance(start, end *layerNode[T]) int {
	if start == nil || end == nil {
		return -1
	}

	if start.Key == end.Key {
		return 0
	}

	// Simple BFS to find shortest path
	visited := make(map[T]bool)
	queue := make([]*layerNode[T], 0)
	distance := make(map[T]int)

	queue = append(queue, start)
	visited[start.Key] = true
	distance[start.Key] = 0

	maxDepth := 10 // Limit search depth to avoid excessive computation

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		currentDist := distance[current.Key]
		if currentDist >= maxDepth {
			continue
		}

		for _, neighbor := range current.neighbors {
			if neighbor == nil {
				continue
			}

			if !visited[neighbor.Key] {
				visited[neighbor.Key] = true
				distance[neighbor.Key] = currentDist + 1
				queue = append(queue, neighbor)

				if neighbor.Key == end.Key {
					return distance[neighbor.Key]
				}
			}
		}
	}

	return -1 // No path found within depth limit
}

// calculateLayerBalance measures how well balanced the layers are.
// It compares the actual layer sizes to the theoretical ideal based on Ml.
// Values closer to 1.0 indicate better balance.
func (a *Analyzer[T]) calculateLayerBalance() float64 {
	if len(a.Graph.layers) <= 1 {
		return 0.0
	}

	ml := a.Graph.Ml
	if ml <= 0 || ml >= 1 {
		return 0
	}

	baseSize := float64(a.Graph.layers[0].size())
	if baseSize == 0 {
		return 0
	}

	var balanceSum float64
	for i := 1; i < len(a.Graph.layers); i++ {
		expectedSize := baseSize * math.Pow(ml, float64(i))
		actualSize := float64(a.Graph.layers[i].size())

		if expectedSize == 0 {
			continue
		}

		// Calculate ratio between actual and expected size
		ratio := actualSize / expectedSize
		if ratio > 1 {
			ratio = 1 / ratio // Ensure ratio is <= 1
		}

		balanceSum += ratio
	}

	return balanceSum / float64(len(a.Graph.layers)-1)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
