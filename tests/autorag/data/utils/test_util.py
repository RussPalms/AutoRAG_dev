import pandas as pd
from langchain_core.documents import Document
from autorag.data.qa.schema import Raw
from autorag.data.utils.util import llama_index_documents_to_raw


def test_llama_index_documents_to_raw():
	# Create sample LlamaIndex documents
	documents = [
		Document(
			page_content="Sample text 1",
			metadata={
				"file_path": "/path/to/file1",
				"page_label": "1",
				"last_modified_datetime": "2023-01-01",
			},
		),
		Document(
			page_content="Sample text 2",
			metadata={
				"file_path": "/path/to/file2",
				"page_label": "2",
				"last_modified_datetime": "2023-01-02",
			},
		),
	]

	# Call the function
	raw = llama_index_documents_to_raw(documents)

	# Check the type of the result
	assert isinstance(raw, Raw)

	# Check the contents of the resulting DataFrame
	expected_data = {
		"texts": ["Sample text 1", "Sample text 2"],
		"path": ["/path/to/file1", "/path/to/file2"],
		"page": ["1", "2"],
		"last_modified_datetime": ["2023-01-01", "2023-01-02"],
	}
	expected_df = pd.DataFrame(expected_data)

	pd.testing.assert_frame_equal(raw.data, expected_df)
