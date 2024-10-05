from autorag.nodes.passagereranker import DashScopeReranker

from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	scores_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)
import pytest


@pytest.fixture
def dashscope_reranker_instance():
	return DashScopeReranker(project_dir=project_dir)


def test_dashscope_reranker_batch_one(dashscope_reranker_instance):
	top_k = 3
	batch = 1
	contents_result, id_result, score_result = dashscope_reranker_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		top_k,
		batch=batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)
