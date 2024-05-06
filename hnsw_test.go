package hnsw

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_maxLevel(t *testing.T) {
	var m int

	m = maxLevel(0.5, 10)
	require.Equal(t, 3, m)

	m = maxLevel(0.5, 1000)
	require.Equal(t, 10, m)
}

type basicPoint struct {
	x float32
}

func (n basicPoint) ID() string {
	return strconv.FormatFloat(float64(n.x), 'f', -1, 32)
}

func (n basicPoint) Embedding() []float32 {
	return []float32{float32(n.x)}
}

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[basicPoint]{
		point: basicPoint{x: 0},
		neighbors: []*layerNode[basicPoint]{
			{
				point: basicPoint{x: 1},
			},
			{
				point: basicPoint{x: 2},
			},
			{
				point: basicPoint{
					x: 3,
				},
				neighbors: []*layerNode[basicPoint]{
					{
						point: basicPoint{x: 3.8},
					},
					{
						point: basicPoint{x: 4.3},
					},
				},
			},
		},
	}

	best := entry.search(2, 2, 4, []float32{4}, EuclideanDistance)

	require.Equal(t, "n3.8", best[0].node.point.ID())
	require.Equal(t, "n4.3", best[1].node.point.ID())
	require.Len(t, best, 2)
}

func TestHNSW_AddSearch(t *testing.T) {
	t.Parallel()

	g := HNSW[basicPoint]{}
	g.Parameters = &DefaultParameters
	g.Parameters.Rng = rand.New(rand.NewSource(0))
	g.Parameters.Distance = EuclideanDistance

	for i := 0; i < 128; i++ {
		g.Add(basicPoint{x: float32(i)})
	}

	require.Equal(t, 7, len(g.layers))
	// Layers should be approximately log2(128) = 7

	// Look for an approximate doubling of the number of nodes in each layer.
	require.Equal(t, 1, g.layers[6].size())
	require.Equal(t, 1, g.layers[5].size())
	require.Equal(t, 4, g.layers[4].size())
	require.Equal(t, 11, g.layers[3].size())
	require.Equal(t, 23, g.layers[2].size())
	require.Equal(t, 66, g.layers[1].size())
	require.Equal(t, 127, g.layers[0].size())

	nearest := g.Search(
		[]float32{64.5},
		4,
	)

	require.Len(t, nearest, 4)
	require.EqualValues(
		t,
		[]basicPoint{
			{x: 64},
			{x: 65},
			{x: 62},
			{x: 63},
		},
		nearest,
	)
}
