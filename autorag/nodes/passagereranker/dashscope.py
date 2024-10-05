import os
from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils import result_to_dataframe
from autorag.utils.util import get_event_loop, process_batch


try:
	import dashscope
except ImportError:
	raise ImportError("DashScope requires `pip install dashscope`")


class DashScopeReranker(BasePassageReranker):
	def __init__(self, project_dir: str, api_key: str = None, *args, **kwargs):
		"""
		Initialize DashScope Rerank node.

		:param project_dir: The project directory path.
		:param api_key: The API key for DashScope rerank.
		You can set it in the environment variable DashScope_API_KEY.
		Or, you can directly set it on the config YAML file using this parameter.
		Default is env variable "DASHSCOPE_API_KEY".
		:param kwargs: Extra arguments that are not affected
		"""
		super().__init__(project_dir)
		if api_key is None:
			self.api_key = os.getenv("DASHSCOPE_API_KEY", None)
			if self.api_key is None:
				raise ValueError(
					"API key is not provided."
					"You can set it as an argument or as an environment variable 'DASHSCOPE_API_KEY'"
				)

		self.dashscope_client = dashscope.TextReRank()

	def __del__(self):
		del self.dashscope_client
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 8)
		model = kwargs.pop("model", "gte-rerank")
		return self._pure(queries, contents, scores, ids, top_k, batch, model)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 8,
		model: str = "gte-rerank",
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with DashScope rerank models.
		You can get the API key from https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key and set it in the environment variable DASHSCOPE_API_KEY.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param scores_list: The list of lists of scores retrieved from the initial ranking
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
		:param model: The model name for DashScope rerank.
		    Default is "gte-rerank".
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		# Run async dashscope_rerank_pure function
		tasks = [
			dashscope_rerank_pure(
				self.dashscope_client, model, query, document, ids, top_k, self.api_key
			)
			for query, document, ids in zip(queries, contents_list, ids_list)
		]
		loop = get_event_loop()
		results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
		content_result = list(map(lambda x: x[0], results))
		id_result = list(map(lambda x: x[1], results))
		score_result = list(map(lambda x: x[2], results))

		return content_result, id_result, score_result


async def dashscope_rerank_pure(
	dashscope_client: dashscope.TextReRank,
	model: str,
	query: str,
	documents: List[str],
	ids: List[str],
	top_k: int,
	api_key: str,
) -> Tuple[List[str], List[str], List[float]]:
	"""
	Rerank a list of contents with DashScope rerank models.

	:param dashscope_client: The DashScope AsyncClient for reranking
	:param model: The model name for DashScope rerank
	:param query: The query to use for reranking
	:param documents: The list of contents to rerank
	:param ids: The list of ids corresponding to the documents
	:param top_k: The number of passages to be retrieved
	:param api_key: The API key for DashScope rerank
	:return: Tuple of lists containing the reranked contents, ids, and scores
	"""
	results = dashscope_client.call(
		model=model,
		top_n=top_k,
		query=query,
		documents=documents,
		return_documents=False,
		api_key=api_key,
	)
	indices = list(map(lambda x: x["index"], results.output.results))
	score_result = list(map(lambda x: x["relevance_score"], results.output.results))
	id_result = list(map(lambda x: ids[x], indices))
	content_result = list(map(lambda x: documents[x], indices))

	return content_result, id_result, score_result
