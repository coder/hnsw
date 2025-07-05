package hnsw

import (
	"cmp"
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

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[int]{
		Node: Node[int]{
			Value: Vector{0},
			Key:   0,
		},
		neighbors: map[int]*layerNode[int]{
			1: {
				Node: Node[int]{
					Value: Vector{1},
					Key:   1,
				},
			},
			2: {
				Node: Node[int]{
					Value: Vector{2},
					Key:   2,
				},
			},
			3: {
				Node: Node[int]{
					Value: Vector{3},
					Key:   3,
				},
				neighbors: map[int]*layerNode[int]{
					4: {
						Node: Node[int]{
							Value: Vector{4},
							Key:   5,
						},
					},
					5: {
						Node: Node[int]{
							Value: Vector{5},
							Key:   5,
						},
					},
				},
			},
		},
	}

	best := entry.search(2, 4, []float32{4}, EuclideanDistance)

	require.Equal(t, 5, best[0].node.Key)
	require.Equal(t, 3, best[1].node.Key)
	require.Len(t, best, 2)
}

func newTestGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        6,
		Distance: EuclideanDistance,
		Ml:       0.5,
		EfSearch: 20,
		Rng:      rand.New(rand.NewSource(0)),
	}
}

func TestGraph_AddSearch(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()

	for i := 0; i < 128; i++ {
		g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
	}

	al := Analyzer[int]{Graph: g}

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
		[]Node[int]{
			{64, Vector{64}},
			{65, Vector{65}},
			{62, Vector{62}},
			{63, Vector{63}},
		},
		nearest,
	)
}

func TestGraph_AddDelete(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g.Add(Node[int]{
			Key:   i,
			Value: Vector{float32(i)},
		})
	}

	require.Equal(t, 128, g.Len())
	an := Analyzer[int]{Graph: g}

	preDeleteConnectivity := an.Connectivity()

	// Delete every even node.
	for i := 0; i < 128; i += 2 {
		ok := g.Delete(i)
		require.True(t, ok)
	}

	require.Equal(t, 64, g.Len())

	postDeleteConnectivity := an.Connectivity()

	// Connectivity should be the same for the lowest layer.
	require.Equal(
		t, preDeleteConnectivity[0],
		postDeleteConnectivity[0],
	)

	t.Run("DeleteNotFound", func(t *testing.T) {
		ok := g.Delete(-1)
		require.False(t, ok)
	})
}

func Benchmark_HSNW(b *testing.B) {
	b.ReportAllocs()

	sizes := []int{100, 1000, 10000}

	// Use this to ensure that complexity is O(log n) where n = h.Len().
	for _, size := range sizes {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			g := Graph[int]{}
			g.Ml = 0.5
			g.Distance = EuclideanDistance
			for i := 0; i < size; i++ {
				g.Add(Node[int]{
					Key:   i,
					Value: Vector{float32(i)},
				})
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

func randFloats(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = rand.Float32()
	}
	return x
}

func Benchmark_HNSW_1536(b *testing.B) {
	b.ReportAllocs()

	g := newTestGraph[int]()
	const size = 1000
	points := make([]Node[int], size)
	for i := 0; i < size; i++ {
		points[i] = Node[int]{
			Key:   i,
			Value: Vector(randFloats(1536)),
		}
		g.Add(points[i])
	}
	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g.Search(
				points[i%size].Value,
				4,
			)
		}
	})
}

func TestGraph_DefaultCosine(t *testing.T) {
	g := NewGraph[int]()
	g.Add(
		Node[int]{Key: 1, Value: Vector{1, 1}},
		Node[int]{Key: 2, Value: Vector{0, 1}},
		Node[int]{Key: 3, Value: Vector{1, -1}},
	)

	neighbors := g.Search(
		[]float32{0.5, 0.5},
		1,
	)

	require.Equal(
		t,
		[]Node[int]{
			{1, Vector{1, 1}},
		},
		neighbors,
	)
}

func TestGraph_RemoveAllNodes(t *testing.T) {
	var vec = []float32{1}

	for i := 0; i < 10; i++ {
		g := NewGraph[int]()
		g.Add(MakeNode(1, vec))
		g.Delete(1)
		g.Add(MakeNode(1, vec))
	}
}
