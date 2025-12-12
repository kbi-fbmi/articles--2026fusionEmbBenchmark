# FBME Bio Library – Tools for Biomedical and Bioinformatics Research

## Getting started for developing

- download sources
```
git clone https://github.com/kbi-fbmi/fmfusions.git
```
- vscode: 
- - open fmfusions.code-workspace - define more projects in one workspace
- - install recomended extensions 

- for each project requirements in projects.toml use uv for adding packages add specific version
```
uv python install [version]
uv sync --python [version]
```
- adding packages pyproject.toml (for dev enviroment)
```
uv add <package>
uv add ---dev <package> 
```
- uv installation
```
pipx install uv 
pip install uv
```

## using library fmlib

- build local 
- check notebooks
- increase version 
- uv sync


## comments git store password
linux
```
git config --global credential.helper store
```
windows
```
git config --global credential.helper wincred