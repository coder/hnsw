package hnsw

import (
	"cmp"
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_maxLevel(t *testing.T) {
	var m int
	var err error

	m, err = maxLevel(0.5, 10)
	require.NoError(t, err)
	require.Equal(t, 4, m)

	m, err = maxLevel(0.5, 1000)
	require.NoError(t, err)
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
		err := g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
		require.NoError(t, err)
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

	nearest, err := g.Search(
		[]float32{64.5},
		4,
	)
	require.NoError(t, err)

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
		err := g.Add(Node[int]{
			Key:   i,
			Value: Vector{float32(i)},
		})
		require.NoError(t, err)
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
			g.M = 16 // Set M to a valid value
			g.Ml = 0.5
			g.Distance = EuclideanDistance
			g.EfSearch = 20                     // Set EfSearch to a valid value
			g.Rng = rand.New(rand.NewSource(0)) // Initialize the random number generator
			for i := 0; i < size; i++ {
				err := g.Add(Node[int]{
					Key:   i,
					Value: Vector{float32(i)},
				})
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()

			b.Run("Search", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_, err := g.Search(
						[]float32{float32(i % size)},
						4,
					)
					if err != nil {
						b.Fatal(err)
					}
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
		err := g.Add(points[i])
		if err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := g.Search(
				points[i%size].Value,
				4,
			)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func TestGraph_DefaultCosine(t *testing.T) {
	g := NewGraph[int]()
	err := g.Add(
		Node[int]{Key: 1, Value: Vector{1, 1}},
		Node[int]{Key: 2, Value: Vector{0, 1}},
		Node[int]{Key: 3, Value: Vector{1, -1}},
	)
	require.NoError(t, err)

	neighbors, err := g.Search(
		[]float32{0.5, 0.5},
		1,
	)
	require.NoError(t, err)

	require.Equal(
		t,
		[]Node[int]{
			{1, Vector{1, 1}},
		},
		neighbors,
	)
}

func Benchmark_Search_vs_ParallelSearch(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	dims := []int{128, 1536}

	for _, size := range sizes {
		for _, dim := range dims {
			b.Run(fmt.Sprintf("Size=%d/Dim=%d/Sequential", size, dim), func(b *testing.B) {
				g := NewGraph[int]()
				g.Distance = CosineDistance

				// Generate random vectors
				for i := 0; i < size; i++ {
					err := g.Add(MakeNode(i, randFloats(dim)))
					if err != nil {
						b.Fatal(err)
					}
				}

				query := randFloats(dim)
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					_, err := g.Search(query, 10)
					if err != nil {
						b.Fatal(err)
					}
				}
			})

			b.Run(fmt.Sprintf("Size=%d/Dim=%d/Parallel", size, dim), func(b *testing.B) {
				g := NewGraph[int]()
				g.Distance = CosineDistance

				// Generate random vectors
				for i := 0; i < size; i++ {
					err := g.Add(MakeNode(i, randFloats(dim)))
					if err != nil {
						b.Fatal(err)
					}
				}

				query := randFloats(dim)
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					_, err := g.ParallelSearch(query, 10, 0) // Use default number of workers
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func Benchmark_LargeGraph_Search(b *testing.B) {
	// Skip this benchmark during regular testing as it's resource-intensive
	if testing.Short() {
		b.Skip("Skipping large graph benchmark in short mode")
	}

	// Create a large graph with high-dimensional vectors
	size := 50000
	dim := 1536

	b.Run(fmt.Sprintf("Size=%d/Dim=%d", size, dim), func(b *testing.B) {
		// Setup phase - create the graph
		b.StopTimer()
		g := NewGraph[int]()
		g.Distance = CosineDistance
		g.EfSearch = 100 // Higher efSearch for better accuracy

		// Generate random vectors
		for i := 0; i < size; i++ {
			err := g.Add(MakeNode(i, randFloats(dim)))
			if err != nil {
				b.Fatal(err)
			}
			if i%10000 == 0 && i > 0 {
				b.Logf("Added %d vectors", i)
			}
		}

		query := randFloats(dim)
		b.StartTimer()

		b.Run("Sequential", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, err := g.Search(query, 10)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("Parallel", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, err := g.ParallelSearch(query, 10, 0) // Use default number of workers
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	})
}

func Benchmark_Delete(b *testing.B) {
	sizes := []int{100, 1000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size=%d", size), func(b *testing.B) {
			// Skip the actual benchmark iterations, just measure the setup time
			b.StopTimer()

			// Create a new graph for each iteration to avoid state issues
			for i := 0; i < b.N; i++ {
				g := NewGraph[int]()
				g.Distance = CosineDistance

				// Generate random vectors
				for j := 0; j < size; j++ {
					err := g.Add(MakeNode(j, randFloats(128)))
					if err != nil {
						b.Fatal(err)
					}
				}

				b.StartTimer()
				// Delete 10% of the nodes
				for j := 0; j < size/10; j++ {
					g.Delete(j)
				}
				b.StopTimer()
			}
		})
	}
}

func TestGraphValidation(t *testing.T) {
	t.Run("ValidConfig", func(t *testing.T) {
		_, err := NewGraphWithConfig[int](16, 0.25, 20, CosineDistance)
		require.NoError(t, err)
	})

	t.Run("InvalidM", func(t *testing.T) {
		_, err := NewGraphWithConfig[int](0, 0.25, 20, CosineDistance)
		require.Error(t, err)
		require.Contains(t, err.Error(), "M must be greater than 0")
	})

	t.Run("InvalidMl", func(t *testing.T) {
		_, err := NewGraphWithConfig[int](16, 0, 20, CosineDistance)
		require.Error(t, err)
		require.Contains(t, err.Error(), "Ml must be between 0 and 1")

		_, err = NewGraphWithConfig[int](16, 1.5, 20, CosineDistance)
		require.Error(t, err)
		require.Contains(t, err.Error(), "Ml must be between 0 and 1")
	})

	t.Run("InvalidEfSearch", func(t *testing.T) {
		_, err := NewGraphWithConfig[int](16, 0.25, 0, CosineDistance)
		require.Error(t, err)
		require.Contains(t, err.Error(), "EfSearch must be greater than 0")
	})

	t.Run("NilDistance", func(t *testing.T) {
		_, err := NewGraphWithConfig[int](16, 0.25, 20, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "Distance function must be set")
	})

	t.Run("InvalidK", func(t *testing.T) {
		g := NewGraph[int]()
		_, err := g.Search([]float32{1, 2, 3}, 0)
		require.Error(t, err)
		require.Contains(t, err.Error(), "k must be greater than 0")

		_, err = g.ParallelSearch([]float32{1, 2, 3}, -1, 4)
		require.Error(t, err)
		require.Contains(t, err.Error(), "k must be greater than 0")
	})
}

func TestThreadSafety(t *testing.T) {
	t.Parallel()

	dims := 3
	numNodes := 100
	numOperations := 1000
	g, err := NewGraphWithConfig[int](16, 0.25, 20, EuclideanDistance)
	require.NoError(t, err)

	// Add initial nodes to the graph
	for i := 0; i < numNodes; i++ {
		vec := make(Vector, dims)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		err := g.Add(MakeNode(i, vec))
		require.NoError(t, err)
	}

	var wg sync.WaitGroup
	wg.Add(numOperations)

	// Run concurrent operations
	for i := 0; i < numOperations; i++ {
		go func(id int) {
			defer wg.Done()

			// Generate a random vector for operations
			vec := make(Vector, dims)
			for j := range vec {
				vec[j] = rand.Float32()
			}

			switch id % 5 {
			case 0: // 20% adds
				err := g.Add(MakeNode(numNodes+id, vec))
				if err != nil {
					t.Logf("Add error: %v", err)
				}
			case 1: // 20% deletes
				g.Delete(id % numNodes)
			default: // 60% searches
				_, err := g.Search(vec, 3)
				if err != nil {
					t.Logf("Search error: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify the graph is still valid
	if err := g.Validate(); err != nil {
		t.Errorf("Graph validation failed after concurrent operations: %v", err)
	}

	// Verify we can still perform operations
	vec := make(Vector, dims)
	for j := range vec {
		vec[j] = rand.Float32()
	}
	_, err = g.Search(vec, 3)
	require.NoError(t, err)

	t.Logf("Graph size after concurrent operations: %d", g.Len())
}
