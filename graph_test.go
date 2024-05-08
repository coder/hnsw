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
	require.Equal(t, 4, m)

	m = maxLevel(0.5, 1000)
	require.Equal(t, 11, m)
}

type basicPoint float32

func (n basicPoint) ID() string {
	return strconv.FormatFloat(float64(n), 'f', -1, 32)
}

func (n basicPoint) Embedding() []float32 {
	return []float32{float32(n)}
}

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[basicPoint]{
		point: basicPoint(0),
		neighbors: map[string]*layerNode[basicPoint]{
			"1": {
				point: basicPoint(1),
			},
			"2": {
				point: basicPoint(2),
			},
			"3": {
				point: basicPoint(3),
				neighbors: map[string]*layerNode[basicPoint]{
					"3.8": {
						point: basicPoint(3.8),
					},
					"4.3": {
						point: basicPoint(4.3),
					},
				},
			},
		},
	}

	best := entry.search(2, 4, []float32{4}, EuclideanDistance)

	require.Equal(t, "3.8", best[0].node.point.ID())
	require.Equal(t, "4.3", best[1].node.point.ID())
	require.Len(t, best, 2)
}

func testGraph[T Embeddable]() *Graph[T] {
	return &Graph[T]{
		M:        6,
		Distance: EuclideanDistance,
		Ml:       0.5,
		EfSearch: 20,
		Rng:      rand.New(rand.NewSource(0)),
	}
}

func TestHNSW_AddSearch(t *testing.T) {
	t.Parallel()

	g := testGraph[basicPoint]()

	for i := 0; i < 128; i++ {
		g.Add(basicPoint(float32(i)))
	}

	al := Analyzer[basicPoint]{Graph: g}

	// Layers should be approximately log2(128) = 7
	// Look for an approximate doubling of the number of nodes in each layer.
	require.Equal(t, []int{
		128,
		67,
		28,
		12,
		6,
		2,
		1,
		1,
	}, al.Topography())

	nearest := g.Search(
		[]float32{64.5},
		4,
	)

	require.Len(t, nearest, 4)
	require.EqualValues(
		t,
		[]basicPoint{
			(64),
			(65),
			(62),
			(63),
		},
		nearest,
	)
}

func TestHNSW_AddDelete(t *testing.T) {
	t.Parallel()

	g := testGraph[basicPoint]()
	for i := 0; i < 128; i++ {
		g.Add(basicPoint(i))
	}

	require.Equal(t, 128, g.Len())
	an := Analyzer[basicPoint]{Graph: g}

	preDeleteConnectivity := an.Connectivity()

	// Delete every even node.
	for i := 0; i < 128; i += 2 {
		g.Delete(basicPoint(i).ID())
	}

	require.Equal(t, 64, g.Len())

	postDeleteConnectivity := an.Connectivity()

	// Connectivity should be the same for the lowest layer.
	require.Equal(
		t, preDeleteConnectivity[0],
		postDeleteConnectivity[0],
	)
}

func Benchmark_HSNW(b *testing.B) {
	b.ReportAllocs()

	sizes := []int{100, 1000, 10000}

	// Use this to ensure that complexity is O(log n) where n = h.Len().
	for _, size := range sizes {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			g := Graph[basicPoint]{}
			for i := 0; i < size; i++ {
				g.Add(basicPoint(i))
			}
			b.ResetTimer()

			b.Run("Search", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					g.Search(
						[]float32{float32(i % size)},
						4,
					)
				}
			})
		})
	}
}

type genericPoint struct {
	id string
	x  []float32
}

func (n genericPoint) ID() string {
	return n.id
}

func (n genericPoint) Embedding() []float32 {
	return n.x
}

func randFloats(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = rand.Float32()
	}
	return x
}

func Benchmark_HNSW_1536(b *testing.B) {
	b.ReportAllocs()

	g := Graph[genericPoint]{}
	const size = 1000
	points := make([]genericPoint, size)
	for i := 0; i < size; i++ {
		points[i] = genericPoint{x: randFloats(1536), id: strconv.Itoa(i)}
		g.Add(points[i])
	}
	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g.Search(
				points[i%size].x,
				4,
			)
		}
	})
}

func TestHNSW_DefaultCosine(t *testing.T) {
	g := NewGraph[Vector]()
	g.Add(
		MakeVector("1", []float32{1, 1}),
		MakeVector("2", []float32{0, 1}),
		MakeVector("3", []float32{1, -1}),
	)

	neighbors := g.Search(
		[]float32{0.5, 0.5},
		1,
	)

	require.Equal(
		t,
		[]Vector{
			MakeVector("1", []float32{1, 1}),
		},
		neighbors,
	)
}
