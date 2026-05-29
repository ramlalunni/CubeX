import os
import re
import subprocess
import glob

def clean_sphinx_boilerplate():
    print("Generating Sphinx .rst files for cubex...")
    # UPDATED PATHS: Output to local "source/api/", input from "../src/" (up one level)
    subprocess.run(["sphinx-apidoc", "-e", "-f", "-o", "source/api/", "../src/"], check=True)

    print("Cleaning up boilerplate text...")
    # UPDATED PATH: Target local "source/api/*.rst"
    for filepath in glob.glob("source/api/*.rst"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # A. Fix the main cubex.rst title
        if os.path.basename(filepath) == "cubex.rst":
            content = re.sub(r'cubex package\n=+', 'CubeX API Reference\n===================', content)

        # Regex helper to dynamically resize the underline
        def fix_title(match):
            title = match.group(1)
            underline = "=" * len(title)
            return f"{title}\n{underline}"

        # B. Strip " package" and " module" from all titles
        content = re.sub(r'^(.+) package\n=+$', fix_title, content, flags=re.MULTILINE)
        content = re.sub(r'^(.+) module\n=+$', fix_title, content, flags=re.MULTILINE)

        # C. Scrub out all the ugly robotic headers
        content = re.sub(r'Module contents\n-+\n', '', content)
        content = re.sub(r'Subpackages\n-+\n', '', content)
        content = re.sub(r'Submodules\n-+\n', '', content)

        # Save the cleaned file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print("✅ API docs generated and auto-cleaned!")

if __name__ == "__main__":
    clean_sphinx_boilerplate()