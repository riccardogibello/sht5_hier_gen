import subprocess
import sys
import os

# Download and install prebuilt sentencepiece wheel for Python 3.13 on Windows
import urllib.request


def _is_gpu_available():
    try:
        if os.name == "nt":  # Windows
            result = subprocess.run(
                ["where", "nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        else:  # Unix or MacOS
            result = subprocess.run(
                ["which", "nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        if result.returncode != 0:
            return False
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_and_activate_venv(venv_name=".venv") -> None:
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", venv_name])

    # Activate virtual environment
    if os.name == "nt":  # Windows
        python_executable = os.path.join(venv_name, "Scripts", "python.exe")
    else:  # Unix or MacOS
        python_executable = os.path.join(venv_name, "bin", "python")

    # Install setuptools and wheel in the virtual environment
    subprocess.run(
        [python_executable, "-m", "pip", "install", "setuptools", "wheel"], check=True
    )

    # Install the requirements from requirements.txt
    subprocess.run(
        [python_executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )

    # Install PyTorch
    if _is_gpu_available():
        print("GPU is available. Installing PyTorch with CUDA support.")
        subprocess.run(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            check=True,
        )
    else:
        print("GPU is not available. Installing PyTorch without CUDA support.")
        subprocess.run(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
            ],
            check=True,
        )

    

    if os.name == "nt": 
        wheel_dir = os.path.join(os.getcwd(), "wheel", "tmp")
        os.makedirs(wheel_dir, exist_ok=True)
        wheel_filename = "sentencepiece-0.2.1-cp313-cp313-win_amd64.whl"
        wheel_path = os.path.join(wheel_dir, wheel_filename)
        wheel_url = (
            "https://github.com/NeoAnthropocene/wheels/raw/f76a39a2c1158b9c8ffcfdc7c0f914f5d2835256/"
            + "sentencepiece-0.2.1-cp313-cp313-win_amd64.whl"
        )

        if os.path.exists(wheel_path):
            print(f"Using existing wheel: {wheel_path}")
        else:
            print(f"Downloading sentencepiece wheel from {wheel_url} ...")
            tmp_path, _ = urllib.request.urlretrieve(wheel_url)
            os.rename(tmp_path, wheel_path)
        subprocess.run([python_executable, "-m", "pip", "install", wheel_path], check=True)
    else:
        subprocess.run(
            [python_executable, "-m", "pip", "install", "sentencepiece"], check=True
        )


if __name__ == "__main__":
    create_and_activate_venv()
