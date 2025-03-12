# HNSW Extensions

This directory contains extensions to the core HNSW implementation, providing additional functionality for specific use cases.

## Available Extensions

### Metadata Extension

The metadata extension allows you to store and retrieve JSON metadata alongside vectors in HNSW graphs. This is useful for applications where you need to associate additional information with each vector, such as product details, document attributes, or user profiles.

**Key features:**

- Store arbitrary JSON metadata with each node
- Retrieve metadata with search results
- Type-safe implementation using Go generics
- Memory-efficient storage
- Support for all HNSW search operations

[Learn more about the Metadata Extension](./meta/README.md)

### Faceted Search Extension

The faceted search extension enables filtering and aggregation of search results based on facets (attributes). This is particularly useful for e-commerce, document search, and other applications where users need to narrow down search results by specific criteria.

**Key features:**

- Filter search results by facet values
- Support for multiple filter types (exact match, range, contains)
- Negative filtering to exclude specific items
- Facet aggregation to count occurrences of facet values
- Efficient implementation that leverages HNSW's fast search capabilities

[Learn more about the Faceted Search Extension](./facets/README.md)

## Running the Examples

To run the examples for all extensions, use the following command:

```bash
cd hnsw-extensions
go run main.go
```

This will demonstrate the usage of each extension with practical examples.

## Creating Your Own Extensions

The HNSW library is designed to be extensible. If you want to create your own extension, consider the following guidelines:

1. Create a new directory for your extension
2. Implement your extension as a separate package
3. Provide a clear API that integrates with the core HNSW functionality
4. Include comprehensive tests and examples
5. Document your extension with a README.md file

For more detailed guidance on creating extensions, see [hnsw-extensions.md](../hnsw-extensions.md).

## Contributing

Contributions to the HNSW extensions are welcome! If you have an idea for a new extension or improvements to existing ones, please open an issue or submit a pull request.
