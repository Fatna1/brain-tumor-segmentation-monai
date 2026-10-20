import modal
import subprocess

app = modal.App("brain-tumor-Segm")
data_volume = modal.Volume.from_name("brain-tumor-data", create_if_missing=True)
weights_volume = modal.Volume.from_name("brain-tumor-weights", create_if_missing=True)


image = (
    modal.Image.debian_slim()
    .pip_install("matplotlib","scipy", "torch", "monai", "requests","tqdm", "monai[nibabel]" )
    .add_local_dir(".", remote_path="/com_neuro_project")  # Include all files in the current directory
)


@app.function(image = image, gpu = "T4", timeout = 60*60*4, volumes = {"/com_neuro_project/monai_dataset": data_volume, "/com_neuro_project/weights" : weights_volume },)
def run_training():
    subprocess.run(["python", "/com_neuro_project/train.py"],cwd = "/com_neuro_project" ,check = True)

@app.function(image = image, gpu = "T4", timeout = 60*60*4, volumes = {"/com_neuro_project/monai_dataset": data_volume, "/com_neuro_project/weights" : weights_volume },)
def run_evaluate():
    subprocess.run(
        ["python", "/com_neuro_project/evaluate.py"],
        cwd="/com_neuro_project",
        check=True,
    )
@app.function(image = image, gpu = "T4", timeout = 60*60*4, volumes = {"/com_neuro_project/monai_dataset": data_volume, "/com_neuro_project/weights" : weights_volume },)
def run_visualize():
    subprocess.run(
        ["python", "/com_neuro_project/visualize.py"],
        cwd="/com_neuro_project",
        check=True,
    )

@app.local_entrypoint()
def main(stage: str ):
    if stage == "train":
        run_training.remote()
    elif stage == "eval":
        run_evaluate.remote()
    elif stage == "vis":
        run_visualize.remote()
    elif stage == "both":
        run_training.remote()
        run_evaluate.remote()