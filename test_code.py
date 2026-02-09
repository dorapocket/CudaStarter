import modal
import os
from pathlib import Path
from datetime import datetime, timezone

app = modal.App()
vol = modal.Volume.from_name("workspace",version=2)
image = modal.Image.from_registry("nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04", add_python="3.12")
out_path = Path("/Users/gl325/Develop/cuda_modal/output")
def download_output(vol, remote_dir: str):
    remote_output_dir = f"{remote_dir}/output"
    local_output_dir = out_path
    local_output_dir.mkdir(parents=True, exist_ok=True)
    entries = vol.listdir(remote_output_dir, recursive=True)
    print("Downloading output files...", entries)
    for entry in entries:
        if entry.type != "file":
            continue
        remote_path = entry.path  # volume 内部路径
        relative_path = remote_path[len(remote_output_dir) + 1 :]
        local_path = local_output_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(remote_path):
                f.write(chunk)
    print("Download finished!")

@app.function(gpu="A100",volumes={"/workspace": vol}, image=image)
def compile_and_run(dir_name: str = "src"):
    import subprocess
    work_dir = Path(f"/workspace/{dir_name}")
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    vol.commit()
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output
    # write to file
    with open(output_dir / "nvidia_smi.txt", "w") as f:
        f.write(output)
    vol.commit()
    print(os.listdir("/workspace/"+dir_name))
    return output                                                        


@app.local_entrypoint()
def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    remote_dir = f"{ts}"
    with vol.batch_upload() as batch:
        batch.put_directory("src", f"/{remote_dir}")
    print(vol.listdir("/", recursive=True))
    print("Uploaded to volume subdir:", remote_dir)
    out = compile_and_run.remote(remote_dir)
    print("Run output:", out)
    download_output(vol, remote_dir)
    vol.remove_file(f"/{remote_dir}", recursive=True)