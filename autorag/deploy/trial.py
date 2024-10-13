import os
import pathlib

import gradio as gr
import pandas as pd
import yaml

from autorag.evaluator import Evaluator
from autorag.deploy import Runner


def display_yaml(file):
	if file is None:
		return "No file uploaded"
	with open(file.name, "r") as f:
		content = yaml.safe_load(f)
	return yaml.dump(content, default_flow_style=False)


def display_parquet(file):
	if file is None:
		return pd.DataFrame()
	df = pd.read_parquet(file.name)
	return df


def check_files(yaml_file, qa_file, corpus_file):
	if yaml_file is not None and qa_file is not None and corpus_file is not None:
		return gr.update(visible=True)
	return gr.update(visible=False)


def run_trial(file, yaml_file, qa_file, corpus_file):
	project_dir = os.path.join(pathlib.PurePath(file.name).parent, "project")
	evaluator = Evaluator(qa_file, corpus_file, project_dir=project_dir)

	evaluator.start_trial(yaml_file, skip_validation=True)
	return "❗Trial Completed❗"


def set_environment_variable(api_name, api_key):
	if api_name and api_key:
		try:
			os.environ[api_name] = api_key
			return "Setting Complete"
		except Exception as e:
			return f"Error setting environment variable: {e}"
	return "API Name or Key is missing"


# Paths to example files
example_yaml = "/Users/kimbwook/PycharmProjects/AutoRAG/sample_config/rag/simple/simple_openai.yaml"
example_qa_parquet = (
	"/Users/kimbwook/PycharmProjects/AutoRAG/tests/resources/qa_data_sample.parquet"
)
example_corpus_parquet = (
	"/Users/kimbwook/PycharmProjects/AutoRAG/tests/resources/corpus_data_sample.parquet"
)

with gr.Blocks() as demo:
	gr.Markdown("### 파일 업로드 UI 및 환경 변수 설정")

	with gr.Tabs() as tabs:
		with gr.Tab("File Upload"):
			with gr.Row() as file_upload_row:
				with gr.Column(scale=3):
					yaml_file = gr.File(
						label="Upload YAML File",
						file_types=[".yaml"],
						file_count="single",
					)
				with gr.Column(scale=7):
					yaml_content = gr.Textbox(label="YAML File Content")
					gr.Markdown("Here is the Sample YAML File. Just click the file ❗")
					gr.Examples(examples=[[example_yaml]], inputs=yaml_file)

			with gr.Row() as qa_upload_row:
				with gr.Column(scale=3):
					qa_file = gr.File(
						label="Upload qa.parquet File",
						file_types=[".parquet"],
						file_count="single",
					)
				with gr.Column(scale=7):
					qa_content = gr.Dataframe(label="QA Parquet File Content")
					gr.Markdown("Here is the Sample QA File. Just click the file ❗")
					gr.Examples(examples=[[example_qa_parquet]], inputs=qa_file)

			with gr.Row() as corpus_upload_row:
				with gr.Column(scale=3):
					corpus_file = gr.File(
						label="Upload corpus.parquet File",
						file_types=[".parquet"],
						file_count="single",
					)
				with gr.Column(scale=7):
					corpus_content = gr.Dataframe(label="Corpus Parquet File Content")
					gr.Markdown(
						"Here is the Sample Corpus File. Just click the file ❗"
					)
					gr.Examples(examples=[[example_corpus_parquet]], inputs=corpus_file)

			run_trial_button = gr.Button("Run Trial", visible=False)
			trial_output = gr.Textbox(label="Trial Output", visible=False)

			# Removed Dashboard button
			web_ui_button = gr.Button("Web UI", visible=False)

			yaml_file.change(display_yaml, inputs=yaml_file, outputs=yaml_content)
			qa_file.change(display_parquet, inputs=qa_file, outputs=qa_content)
			corpus_file.change(
				display_parquet, inputs=corpus_file, outputs=corpus_content
			)

			yaml_file.change(
				check_files,
				inputs=[yaml_file, qa_file, corpus_file],
				outputs=run_trial_button,
			)
			qa_file.change(
				check_files,
				inputs=[yaml_file, qa_file, corpus_file],
				outputs=run_trial_button,
			)
			corpus_file.change(
				check_files,
				inputs=[yaml_file, qa_file, corpus_file],
				outputs=run_trial_button,
			)

			run_trial_button.click(
				lambda: (
					gr.update(visible=False),
					gr.update(visible=False),
					gr.update(visible=False),
					gr.update(visible=True),
					gr.update(visible=True),
				),
				outputs=[
					file_upload_row,
					qa_upload_row,
					corpus_upload_row,
					trial_output,
					web_ui_button,
				],
			)
			run_trial_button.click(
				run_trial,
				inputs=[yaml_file, yaml_file, qa_file, corpus_file],
				outputs=trial_output,
			)

			# Connect Web UI button to switch to the chat tab
			web_ui_button.click(
				lambda: gr.update(selected="Chat"), inputs=None, outputs=None
			)

		with gr.Tab("Environment Variables"):
			gr.Markdown("### 환경 변수 설정")
			with gr.Row():  # Arrange horizontally
				with gr.Column(scale=3):
					api_name = gr.Textbox(
						label="Environment Variable Name",
						type="text",
						placeholder="Enter your Environment Variable Name",
					)
				with gr.Column(scale=7):
					api_key = gr.Textbox(
						label="API Key",
						type="password",
						placeholder="Enter your API Key",
					)

			set_env_button = gr.Button("Set Environment Variable")
			env_output = gr.Textbox(
				label="Environment Variable Status", interactive=False
			)

			set_env_button.click(
				set_environment_variable, inputs=[api_name, api_key], outputs=env_output
			)

		# New Chat Tab
		with gr.Tab("Chat") as chat_tab:
			gr.Markdown("### Compare Chat Models")

			question_input = gr.Textbox(
				label="Your Question", placeholder="Type your question here..."
			)

			with gr.Row():
				# Left Chatbox (Default YAML)
				with gr.Column():
					gr.Markdown("#### Default YAML Chat")
					default_chatbox = gr.Chatbot(label="Default YAML Conversation")

				# Right Chatbox (Custom YAML)
				with gr.Column():
					gr.Markdown("#### Custom YAML Chat")
					custom_chatbox = gr.Chatbot(label="Custom YAML Conversation")

			def run_both_chats(file, question):
				# Default YAML Runner
				yaml_path = "/Users/kimbwook/PycharmProjects/AutoRAG/sample_config/rag/extracted_sample.yaml"
				project_dir = os.path.join(
					pathlib.PurePath(file.name).parent, "project"
				)
				default_runner = Runner.from_yaml(yaml_path, project_dir)
				default_answer = default_runner.run(question)

				# Custom YAML Runner
				trial_dir = os.path.join(project_dir, "0")
				custom_runner = Runner.from_trial_folder(trial_dir)
				custom_answer = custom_runner.run(question)

				# Return responses for both chatboxes
				return [(question, default_answer)], [(question, custom_answer)]

			question_input.submit(
				run_both_chats,
				inputs=[yaml_file, question_input],
				outputs=[default_chatbox, custom_chatbox],
			)

demo.launch()
