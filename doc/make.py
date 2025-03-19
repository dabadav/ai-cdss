import os
import shutil
import subprocess

def clean_docs():
    """Removes old documentation before regenerating."""
    build_path = "doc/build"
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
        print("Removed old documentation.")

def generate_apidoc():
    """Automatically generates Sphinx documentation from Python docstrings."""
    source_dir = "doc/source/api"
    module_dir = "src/ai_cdss"  # Change this to your Python package directory
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    
    subprocess.run(["sphinx-apidoc", "-o", source_dir, module_dir, "--force"])
    print("Generated API documentation.")

def build_docs():
    """Builds HTML documentation."""
    subprocess.run(["sphinx-build", "-b", "html", "doc/source", "doc/build"])

    # index_path = "doc/build/index.html"
    # ai_cdss_path = "doc/build/ai_cdss.html"

    # if os.path.exists(ai_cdss_path):
    #     shutil.copy(ai_cdss_path, index_path)
    #     print("index.html successfully replaced with ai_cdss.html!")

    print("Documentation successfully built! Open doc/build/index.html in a browser.")

if __name__ == "__main__":
    clean_docs()
    generate_apidoc()
    build_docs()
