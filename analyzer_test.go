package hnsw

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnalyzer_QualityMetrics(t *testing.T) {
	// Create a test graph
	g := newTestGraph[int]()

	// Empty graph should return default metrics
	analyzer := Analyzer[int]{Graph: g}
	metrics := analyzer.QualityMetrics()

	assert.Equal(t, 0, metrics.NodeCount)
	assert.Equal(t, 0.0, metrics.AvgConnectivity)
	assert.Equal(t, 0.0, metrics.ConnectivityStdDev)

	// Add nodes to the graph
	for i := 0; i < 100; i++ {
		err := g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
		require.NoError(t, err)
	}

	// Get metrics for populated graph
	metrics = analyzer.QualityMetrics()

	// Basic assertions
	assert.Equal(t, 100, metrics.NodeCount)
	assert.Greater(t, metrics.AvgConnectivity, 0.0)
	assert.GreaterOrEqual(t, metrics.ConnectivityStdDev, 0.0)
	assert.GreaterOrEqual(t, metrics.GraphHeight, 1)

	// Layer balance should be between 0 and 1
	assert.GreaterOrEqual(t, metrics.LayerBalance, 0.0)
	assert.LessOrEqual(t, metrics.LayerBalance, 1.0)

	// Distortion ratio should be positive or zero
	assert.GreaterOrEqual(t, metrics.DistortionRatio, 0.0)
}

func TestAnalyzer_EstimateGraphDistance(t *testing.T) {
	// Create a simple graph with known structure
	g := newTestGraph[int]()

	// Add nodes in a line: 0 -> 1 -> 2 -> 3
	for i := 0; i < 4; i++ {
		err := g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
		require.NoError(t, err)
	}

	analyzer := Analyzer[int]{Graph: g}

	// Get nodes from the base layer
	baseLayer := g.layers[0]
	node0 := baseLayer.nodes[0]
	node1 := baseLayer.nodes[1]
	node3 := baseLayer.nodes[3]

	// Test distances
	assert.Equal(t, 0, analyzer.estimateGraphDistance(node0, node0), "Distance to self should be 0")

	// Due to the probabilistic nature of the graph, we can't guarantee exact distances
	// But we can check that distances are reasonable
	dist01 := analyzer.estimateGraphDistance(node0, node1)
	assert.GreaterOrEqual(t, dist01, 0, "Distance should be non-negative")

	dist03 := analyzer.estimateGraphDistance(node0, node3)
	assert.GreaterOrEqual(t, dist03, 0, "Distance should be non-negative")

	// Test with nil nodes
	assert.Equal(t, -1, analyzer.estimateGraphDistance(nil, node1), "Distance with nil node should be -1")
	assert.Equal(t, -1, analyzer.estimateGraphDistance(node0, nil), "Distance with nil node should be -1")
}

func TestAnalyzer_ConnectivityMetrics(t *testing.T) {
	// Create a test graph
	g := newTestGraph[int]()

	// Add nodes to the graph
	for i := 0; i < 50; i++ {
		err := g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
		require.NoError(t, err)
	}

	analyzer := Analyzer[int]{Graph: g}

	// Test average connectivity
	avgConn := analyzer.averageConnectivity()
	assert.GreaterOrEqual(t, avgConn, 0.0, "Average connectivity should be non-negative")
	assert.LessOrEqual(t, avgConn, float64(g.M), "Average connectivity should not exceed M")

	// Test connectivity standard deviation
	stdDev := analyzer.connectivityStdDev()
	assert.GreaterOrEqual(t, stdDev, 0.0, "Standard deviation should be non-negative")
}

func TestAnalyzer_LayerBalance(t *testing.T) {
	// Create a test graph
	g := newTestGraph[int]()
	g.Ml = 0.5 // Set level generation factor

	// Add enough nodes to create multiple layers
	for i := 0; i < 100; i++ {
		err := g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
		require.NoError(t, err)
	}

	analyzer := Analyzer[int]{Graph: g}

	// Test layer balance
	balance := analyzer.calculateLayerBalance()
	assert.GreaterOrEqual(t, balance, 0.0, "Layer balance should be non-negative")
	assert.LessOrEqual(t, balance, 1.0, "Layer balance should not exceed 1.0")

	// Check topography
	topo := analyzer.Topography()
	assert.GreaterOrEqual(t, len(topo), 2, "Should have at least 2 layers")

	// Each layer should be approximately half the size of the previous layer (Ml = 0.5)
	for i := 1; i < len(topo); i++ {
		if topo[i-1] > 0 {
			ratio := float64(topo[i]) / float64(topo[i-1])
			// Allow for some variance due to randomness
			assert.LessOrEqual(t, ratio, 1.0, "Higher layer should not be larger than lower layer")
		}
	}
}

func TestAnalyzer_DistortionRatio(t *testing.T) {
	// Create a test graph
	g := newTestGraph[int]()

	// Add nodes to the graph
	for i := 0; i < 20; i++ {
		err := g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
		require.NoError(t, err)
	}

	analyzer := Analyzer[int]{Graph: g}

	// Test distortion ratio
	distortion := analyzer.calculateDistortionRatio()
	assert.GreaterOrEqual(t, distortion, 0.0, "Distortion ratio should be non-negative")
}

func TestAnalyzer_EmptyGraph(t *testing.T) {
	// Create an empty graph
	g := newTestGraph[int]()
	analyzer := Analyzer[int]{Graph: g}

	// Test all metrics with empty graph
	assert.Equal(t, 0.0, analyzer.averageConnectivity())
	assert.Equal(t, 0.0, analyzer.connectivityStdDev())
	assert.Equal(t, 0.0, analyzer.calculateDistortionRatio())
	assert.Equal(t, 0.0, analyzer.calculateLayerBalance()) // Default for empty graph

	metrics := analyzer.QualityMetrics()
	assert.Equal(t, 0, metrics.NodeCount)
	assert.Equal(t, 0.0, metrics.AvgConnectivity)
	assert.Equal(t, 0.0, metrics.ConnectivityStdDev)
	assert.Equal(t, 0.0, metrics.DistortionRatio)
	assert.Equal(t, 0.0, metrics.LayerBalance)
	assert.Equal(t, 0, metrics.GraphHeight)
}
