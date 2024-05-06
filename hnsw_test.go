package hnsw

import (
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
	id string
	x  float32
}

func (n basicPoint) ID() string {
	return n.id
}

func (n basicPoint) Embedding() []float32 {
	return []float32{float32(n.x)}
}

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[basicPoint]{
		point: basicPoint{id: "entry", x: 0},
		neighbors: []*layerNode[basicPoint]{
			{
				point: basicPoint{id: "n1", x: 1},
			},
			{
				point: basicPoint{id: "n2", x: 2},
			},
			{
				point: basicPoint{
					id: "n3", x: 3,
				},
				neighbors: []*layerNode[basicPoint]{
					{
						point: basicPoint{id: "n3.8", x: 3.8},
					},
					{
						point: basicPoint{id: "n4.3", x: 4.3},
					},
				},
			},
		},
	}

	target := basicPoint{id: "target", x: 4}

	best := entry.search(2, target, EuclideanDistance)
	if best.point.ID() != "n3.8" {
		t.Errorf("unexpected best node: %s", best.point.ID())
	}
}
