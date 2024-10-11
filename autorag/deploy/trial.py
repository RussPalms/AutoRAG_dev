import gradio as gr
import pandas as pd
import yaml

from autorag.evaluator import Evaluator


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
	# Check if all files are uploaded
	if yaml_file is not None and qa_file is not None and corpus_file is not None:
		return gr.update(visible=True)
	return gr.update(visible=False)


def run_trial(file, qa_file, corpus_file, yaml_file):
	project_dir = ""
	evaluator = Evaluator(qa_file, corpus_file, project_dir=project_dir)
	return evaluator.start_trial(yaml_file, skip_validation=False)


# Paths to example files
example_yaml = "/Users/kimbwook/PycharmProjects/AutoRAG/tests/resources/simple.yaml"
example_qa_parquet = (
	"/Users/kimbwook/PycharmProjects/AutoRAG/tests/resources/qa_data_sample.parquet"
)
example_corpus_parquet = (
	"/Users/kimbwook/PycharmProjects/AutoRAG/tests/resources/corpus_data_sample.parquet"
)

with gr.Blocks() as demo:
	gr.Markdown("### 파일 업로드 UI")

	with gr.Row() as file_upload_row:
		with gr.Column(scale=3):
			yaml_file = gr.File(
				label="Upload YAML File", file_types=[".yaml"], file_count="single"
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
			gr.Markdown("Here is the Sample Corpus File. Just click the file ❗")
			gr.Examples(examples=[[example_corpus_parquet]], inputs=corpus_file)

	# Add the Run Trial button, initially hidden
	run_trial_button = gr.Button("Run Trial", visible=False)
	trial_output = gr.Textbox(label="Trial Output", visible=False)

	# Set up file change events
	yaml_file.change(display_yaml, inputs=yaml_file, outputs=yaml_content)
	qa_file.change(display_parquet, inputs=qa_file, outputs=qa_content)
	corpus_file.change(display_parquet, inputs=corpus_file, outputs=corpus_content)

	# Check if all files are uploaded to show the Run Trial button
	yaml_file.change(
		check_files, inputs=[yaml_file, qa_file, corpus_file], outputs=run_trial_button
	)
	qa_file.change(
		check_files, inputs=[yaml_file, qa_file, corpus_file], outputs=run_trial_button
	)
	corpus_file.change(
		check_files, inputs=[yaml_file, qa_file, corpus_file], outputs=run_trial_button
	)

	# Run trial button click event
	run_trial_button.click(
		lambda: (
			gr.update(visible=False),
			gr.update(visible=False),
			gr.update(visible=False),
			gr.update(visible=True),
		),
		outputs=[file_upload_row, qa_upload_row, corpus_upload_row, trial_output],
	)
	run_trial_button.click(run_trial, outputs=trial_output)

demo.launch()
