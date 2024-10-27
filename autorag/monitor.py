from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Define Prometheus metrics for query expansion
query_expansion_execution_counter = metrics.counter(
	"query_expansion_module_executions",
	"Number of times each query expansion module is executed",
	labels={"module_name": lambda: "default"},
)

query_expansion_progress_gauge = metrics.gauge(
	"query_expansion_module_progress",
	"Progress percentage of each query expansion module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for retrieval
retrieval_execution_counter = metrics.counter(
	"retrieval_module_executions",
	"Number of times each retrieval module is executed",
	labels={"module_name": lambda: "default"},
)

retrieval_progress_gauge = metrics.gauge(
	"retrieval_module_progress",
	"Progress percentage of each retrieval module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for Passage Augmenter
passage_augmenter_execution_counter = metrics.counter(
	"passage_augmenter_module_executions",
	"Number of times each passage augmenter module is executed",
	labels={"module_name": lambda: "default"},
)

passage_augmenter_progress_gauge = metrics.gauge(
	"passage_augmenter_module_progress",
	"Progress percentage of each passage augmenter module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for Passage Reranker
passage_reranker_execution_counter = metrics.counter(
	"passage_reranker_module_executions",
	"Number of times each passage reranker module is executed",
	labels={"module_name": lambda: "default"},
)

passage_reranker_progress_gauge = metrics.gauge(
	"passage_reranker_module_progress",
	"Progress percentage of each passage reranker module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for Passage Filter
passage_filter_execution_counter = metrics.counter(
	"passage_filter_module_executions",
	"Number of times each passage filter module is executed",
	labels={"module_name": lambda: "default"},
)

passage_filter_progress_gauge = metrics.gauge(
	"passage_filter_module_progress",
	"Progress percentage of each passage filter module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for Passage Compressor
passage_compressor_execution_counter = metrics.counter(
	"passage_compressor_module_executions",
	"Number of times each passage compressor module is executed",
	labels={"module_name": lambda: "default"},
)

passage_compressor_progress_gauge = metrics.gauge(
	"passage_compressor_module_progress",
	"Progress percentage of each passage compressor module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for Prompt Maker
prompt_maker_execution_counter = metrics.counter(
	"prompt_maker_module_executions",
	"Number of times each prompt maker module is executed",
	labels={"module_name": lambda: "default"},
)

prompt_maker_progress_gauge = metrics.gauge(
	"prompt_maker_module_progress",
	"Progress percentage of each prompt maker module",
	labels={"module_name": lambda: "default"},
)

# Define Prometheus metrics for Generator
generator_execution_counter = metrics.counter(
	"generator_module_executions",
	"Number of times each generator module is executed",
	labels={"module_name": lambda: "default"},
)

generator_progress_gauge = metrics.gauge(
	"generator_module_progress",
	"Progress percentage of each generator module",
	labels={"module_name": lambda: "default"},
)


@app.route("/")
def hello():
	return "Hello, World!"


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)
