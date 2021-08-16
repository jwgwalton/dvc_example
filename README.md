# Using DVC and CML for sharing ML projects across teams

This project is motivated as an example to show how to share datasets across teams in the AI function at Corsearch. 

Requirements:
- Versioned datasets stored in an accessible remote storage
- Ability to identify the best of breed solution to the problem

## What is this project 
This repository contains a sample project using [CML](https://cml.dev/) with [DVC](https://dvc.org/) to push/pull data from cloud storage and track model metrics. When a pull request is made in this repository, the following will occur:

    - Github will deploy a runner machine with a specified CML Docker environment
    - DVC will pull data from cloud storage
    - The runner will execute a workflow to train a ML model (python train.py)
    - A visual CML report about the model performance with DVC metrics will be returned as a comment in the pull request
 
### Reproduceability
ML = Data + Code. 

[DVC](https://dvc.org/) works as Git for Data so we have full lineage of what code + data produces what results. 
For separate teams to work on the same problems we need to be able to assess the output

DVC saves the md5 hashes of the data files in git. An example of a dvc file below

```bash
outs:
- md5: 79b2176dd366f3be286780a501207603
  size: 990848
  path: X_train.npy
``` 

As this dvc file is checked into version control we know what data and outputs relate to which version of code.

DVC pipelines provide md5 hashes of all parts of an ML workflow including data and code so we have full reproducability.``


## Cloning this project
Note that if you clone this project, you will have to configure your own DVC storage and credentials for the example. We suggest the following procedure:

1. Fork the repository and clone to your local workstation. 
2. Run `python src/prepare_data.py` to generate your own copy of the dataset. 
3. Initialise DVC `dvc init` and setup the remote storage `dvc remote add storage <your bucket>`, `dvc remote default storage`
4. Push your data to DVC storage `dvc add data/raw`, `dvc push`
5. `git add`, `commit` and `push` to push your DVC configuration to GitHub.
6. Add your storage credentials as repository secrets.
7. Copy the workflow file `.github/workflows/cml.yml` from this repository to your fork. By default, workflow files are not copied in forks. When you commit this file to your repository, the first workflow should be initiated. 
