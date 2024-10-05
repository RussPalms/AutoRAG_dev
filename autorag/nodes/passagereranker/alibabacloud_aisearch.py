import os
from typing import List, Tuple
import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils import result_to_dataframe

try:
	from alibabacloud_tea_openapi.models import Config as AISearchConfig
	from alibabacloud_searchplat20240529.client import Client
except ImportError:
	raise ImportError(
		"Could not import alibabacloud_searchplat20240529 python package. "
		"Please install it with `pip install alibabacloud-searchplat20240529`."
	)


class AlibabaCloudAISearchReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		api_key: str = None,
		endpoint: str = None,
		*args,
		**kwargs,
	):
		super().__init__(project_dir)
		if api_key is None:
			api_key = os.getenv("AISEARCH_API_KEY", None)
			if api_key is None:
				raise ValueError(
					"API key is not provided."
					"You can set it as an argument or as an environment variable 'JINAAI_API_KEY'"
				)
		if endpoint is None:
			endpoint = os.getenv("AISEARCH_ENDPOINT", None)
			if endpoint is None:
				raise ValueError(
					"API key is not provided."
					"You can set it as an argument or as an environment variable 'JINAAI_API_KEY'"
				)

		config = AISearchConfig(
			bearer_token=api_key,
			endpoint=endpoint,
			protocol="http",
		)

		self.aisearch_client = Client(config=config)

	def __del__(self):
		del self.aisearch_client
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		return self._pure(queries, contents, ids, top_k)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		pass
