package hnsw

import "math"

// DistanceFunc is a function that computes the distance between two vectors.
type DistanceFunc func(a, b []float32) float32

// CosineDistance computes the cosine distance between two vectors.
func CosineDistance(a, b []float32) float32 {
	var dotProduct float32 = 0
	var normA float32 = 0
	var normB float32 = 0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0 // Cosine similarity is not defined when one vector is zero.
	}

	sim := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1 - sim
}

// EuclideanDistance computes the Euclidean distance between two vectors.
func EuclideanDistance(a, b []float32) float32 {
	var sum float32 = 0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}
