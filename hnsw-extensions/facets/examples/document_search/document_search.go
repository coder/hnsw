package examples

import (
	"fmt"
	"log"
	"time"

	"github.com/TFMV/hnsw"
	"github.com/TFMV/hnsw/hnsw-extensions/facets"
)

// Document represents a document with various attributes.
type Document struct {
	ID       string
	Title    string
	Content  string
	Author   string
	Date     time.Time
	Category string
	Tags     []string
	Vector   []float32 // Embedding vector
}

// DocumentSearch demonstrates how to use faceted search for document search.
func DocumentSearch() {
	// Create a new HNSW graph
	graph := hnsw.NewGraph[string]()

	// Create a facet store
	store := facets.NewMemoryFacetStore[string]()

	// Create a faceted graph
	facetedGraph := facets.NewFacetedGraph(graph, store)

	// Add some documents
	documents := []Document{
		{
			ID:       "doc1",
			Title:    "Introduction to Machine Learning",
			Content:  "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
			Author:   "John Smith",
			Date:     parseDate("2023-01-15"),
			Category: "Technology",
			Tags:     []string{"machine learning", "AI", "technology"},
			Vector:   []float32{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		{
			ID:       "doc2",
			Title:    "Advanced Deep Learning Techniques",
			Content:  "This paper explores advanced techniques in deep learning, including transformers and attention mechanisms.",
			Author:   "Jane Doe",
			Date:     parseDate("2023-03-20"),
			Category: "Technology",
			Tags:     []string{"deep learning", "AI", "neural networks"},
			Vector:   []float32{0.2, 0.3, 0.4, 0.5, 0.6},
		},
		{
			ID:       "doc3",
			Title:    "Climate Change Impact on Biodiversity",
			Content:  "This study examines how climate change affects biodiversity in various ecosystems.",
			Author:   "David Johnson",
			Date:     parseDate("2023-02-10"),
			Category: "Environment",
			Tags:     []string{"climate change", "biodiversity", "environment"},
			Vector:   []float32{0.5, 0.6, 0.7, 0.8, 0.9},
		},
		{
			ID:       "doc4",
			Title:    "Sustainable Energy Solutions",
			Content:  "An overview of sustainable energy solutions for reducing carbon emissions.",
			Author:   "Sarah Williams",
			Date:     parseDate("2023-04-05"),
			Category: "Environment",
			Tags:     []string{"sustainable energy", "renewable", "environment"},
			Vector:   []float32{0.4, 0.5, 0.6, 0.7, 0.8},
		},
		{
			ID:       "doc5",
			Title:    "The Future of Artificial Intelligence",
			Content:  "This article discusses the future trends and potential impacts of artificial intelligence.",
			Author:   "John Smith",
			Date:     parseDate("2023-05-12"),
			Category: "Technology",
			Tags:     []string{"AI", "future", "technology"},
			Vector:   []float32{0.15, 0.25, 0.35, 0.45, 0.55},
		},
	}

	// Add documents to the faceted graph
	for _, doc := range documents {
		// Create HNSW node
		node := hnsw.MakeNode(doc.ID, doc.Vector)

		// Create facets
		docFacets := []facets.Facet{
			facets.NewBasicFacet("title", doc.Title),
			facets.NewBasicFacet("author", doc.Author),
			facets.NewBasicFacet("date", doc.Date),
			facets.NewBasicFacet("category", doc.Category),
		}

		// Add tags as separate facets
		for _, tag := range doc.Tags {
			docFacets = append(docFacets, facets.NewBasicFacet("tag", tag))
		}

		// Create faceted node
		facetedNode := facets.NewFacetedNode(node, docFacets)

		// Add to faceted graph
		err := facetedGraph.Add(facetedNode)
		if err != nil {
			log.Fatalf("Failed to add document %s: %v", doc.ID, err)
		}
	}

	// Example 1: Search for documents similar to "The Future of Artificial Intelligence" with author filter
	fmt.Println("Example 1: Search for documents similar to 'The Future of Artificial Intelligence' with author filter")
	queryVector := documents[4].Vector // The Future of AI vector
	authorFilter := facets.NewEqualityFilter("author", "John Smith")

	results, err := facetedGraph.Search(
		queryVector,
		[]facets.FacetFilter{authorFilter},
		3,
		2,
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		docID := result.Node.Key
		doc := findDocumentByID(documents, docID)
		fmt.Printf("%d. %s - by %s (%s)\n", i+1, doc.Title, doc.Author, doc.Category)
	}
	fmt.Println()

	// Example 2: Search for technology documents with date range filter
	fmt.Println("Example 2: Search for technology documents with date range filter")
	categoryFilter := facets.NewEqualityFilter("category", "Technology")

	// Create a custom date range filter
	startDate := parseDate("2023-03-01")
	endDate := parseDate("2023-06-01")
	dateFilter := &DateRangeFilter{
		name:      "date",
		startDate: startDate,
		endDate:   endDate,
	}

	results, err = facetedGraph.Search(
		queryVector,
		[]facets.FacetFilter{categoryFilter, dateFilter},
		3,
		2,
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		docID := result.Node.Key
		doc := findDocumentByID(documents, docID)
		fmt.Printf("%d. %s - by %s (%s, %s)\n", i+1, doc.Title, doc.Author, doc.Category, doc.Date.Format("2006-01-02"))
	}
	fmt.Println()

	// Example 3: Search with negative example (avoid climate change documents)
	fmt.Println("Example 3: Search with negative example (avoid climate change documents)")
	negativeVector := documents[2].Vector // Climate Change Impact vector

	results, err = facetedGraph.SearchWithNegative(
		queryVector,
		negativeVector,
		[]facets.FacetFilter{},
		3,
		0.7,
		2,
	)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Results:")
	for i, result := range results {
		docID := result.Node.Key
		doc := findDocumentByID(documents, docID)
		fmt.Printf("%d. %s - by %s (%s)\n", i+1, doc.Title, doc.Author, doc.Category)
	}
	fmt.Println()

	// Example 4: Get facet aggregations
	fmt.Println("Example 4: Get facet aggregations")
	aggregations, err := facetedGraph.GetFacetAggregations(
		queryVector,
		[]facets.FacetFilter{},
		[]string{"category", "author", "tag"},
		5,
		1,
	)
	if err != nil {
		log.Fatalf("Aggregation failed: %v", err)
	}

	fmt.Println("Category aggregations:")
	for value, count := range aggregations["category"].Values {
		fmt.Printf("- %s: %d documents\n", value, count)
	}

	fmt.Println("Author aggregations:")
	for value, count := range aggregations["author"].Values {
		fmt.Printf("- %s: %d documents\n", value, count)
	}

	fmt.Println("Tag aggregations:")
	for value, count := range aggregations["tag"].Values {
		fmt.Printf("- %s: %d documents\n", value, count)
	}
}

// DateRangeFilter is a custom filter that matches dates within a range.
type DateRangeFilter struct {
	name      string
	startDate time.Time
	endDate   time.Time
}

// Name returns the name of the facet this filter applies to.
func (f *DateRangeFilter) Name() string {
	return f.name
}

// Matches checks if a date facet value is within the range.
func (f *DateRangeFilter) Matches(value interface{}) bool {
	date, ok := value.(time.Time)
	if !ok {
		return false
	}
	return (date.Equal(f.startDate) || date.After(f.startDate)) &&
		(date.Equal(f.endDate) || date.Before(f.endDate))
}

// parseDate parses a date string in the format "YYYY-MM-DD".
func parseDate(dateStr string) time.Time {
	date, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		log.Fatalf("Failed to parse date %s: %v", dateStr, err)
	}
	return date
}

// findDocumentByID finds a document by its ID.
func findDocumentByID(documents []Document, id string) Document {
	for _, doc := range documents {
		if doc.ID == id {
			return doc
		}
	}
	return Document{}
}
