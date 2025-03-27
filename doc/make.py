import os
import shutil
import subprocess

def clean_docs():
    """Removes old documentation before regenerating."""
    build_path = "doc/build/html"
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
        print("Removed old documentation.")

def build_docs():
    """Builds HTML documentation."""
    output_dir = "doc/build/html"
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["sphinx-build", "-b", "html", "doc/source", output_dir])
    print("Documentation successfully built! Open doc/build/html/index.html in a browser.")

if __name__ == "__main__":
    clean_docs()
    build_docs()
